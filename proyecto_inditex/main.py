import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_CATALOGO = os.path.join(RUTA_BASE, "data", "catalog_images", "*.jpg")
CARPETA_BUNDLES = os.path.join(RUTA_BASE, "data", "raw_images")
ARCHIVO_TEST = os.path.join(RUTA_BASE, "bundles_product_match_test.csv")
ARCHIVO_SALIDA = os.path.join(RUTA_BASE, "resultado_color.csv")
ARCHIVO_VECTORES = os.path.join(RUTA_BASE, "vectores_catalogo_27k.pt")
ARCHIVO_COLORES = os.path.join(RUTA_BASE, "colores_catalogo.pt")
RUTA_MODELO_YOLO = os.path.join(RUTA_BASE, "deepfashion2_yolov8s-seg.pt")
ARCHIVO_METADATOS = os.path.join(RUTA_BASE, "metadata", "product_dataset.csv")

UMBRAL = 0.28
TOP_K = 5
UMBRAL_COLOR = 60
BATCH_SIZE = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

model_yolo = YOLO(RUTA_MODELO_YOLO)
model_clip = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
processor_clip = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")


def clasificar_prenda(nombre_clase):
    nombre = nombre_clase.lower()
    if any(p in nombre for p in ['shirt', 'outwear', 'vest', 'sling', 'dress', 'top', 'coat', 'jacket', 'blazer', 'sweater']):
        return 'top'
    elif any(p in nombre for p in ['shorts', 'trousers', 'skirt', 'pants', 'jeans', 'leggings']):
        return 'bottom'
    elif any(p in nombre for p in ['shoe', 'sneaker', 'boot', 'sandal', 'heel', 'loafer']):
        return 'shoe'
    return 'accesories'


def color_dominante(img_pil):
    img = img_pil.resize((64, 64)).convert('RGB')
    arr = np.array(img).reshape(-1, 3).astype(float)
    mask = ~((arr[:, 0] > 220) & (arr[:, 1] > 220) & (arr[:, 2] > 220))
    arr = arr[mask]
    if len(arr) < 10:
        return np.array([128, 128, 128])
    np.random.seed(0)
    centroids = arr[np.random.choice(len(arr), 3, replace=False)]
    for _ in range(10):
        dists = np.linalg.norm(arr[:, None] - centroids[None], axis=2)
        labels = np.argmin(dists, axis=1)
        nuevos = np.array([arr[labels == k].mean(axis=0) if (labels == k).any() else centroids[k] for k in range(3)])
        if np.allclose(centroids, nuevos):
            break
        centroids = nuevos
    conteos = [(labels == k).sum() for k in range(3)]
    return centroids[np.argmax(conteos)].astype(int)


def colores_similares(c1, c2, umbral=UMBRAL_COLOR):
    return np.linalg.norm(c1.astype(float) - c2.astype(float)) < umbral


def obtener_recortes_yolo(ruta_bundle):
    resultados = model_yolo(ruta_bundle, verbose=False, conf=0.3, iou=0.5)
    img = Image.open(ruta_bundle).convert("RGB")
    w, h = img.size
    recortes = []

    if resultados[0].boxes:
        for box, cls, conf in zip(resultados[0].boxes.xyxy, resultados[0].boxes.cls, resultados[0].boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            if (x2 - x1) < 30 or (y2 - y1) < 30:
                continue
            if ((x2 - x1) * (y2 - y1)) < (w * h * 0.01):
                continue

            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(w, x2 + 10)
            y2 = min(h, y2 + 10)

            recorte = img.crop((x1, y1, x2, y2))
            recortes.append({
                "imagen": recorte,
                "categoria": clasificar_prenda(model_yolo.names[int(cls)]),
                "color": color_dominante(recorte),
                "confianza": float(conf)
            })

    return recortes


def obtener_embedding(imagen_pil):
    inputs = processor_clip(images=imagen_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model_clip.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
    return (features / features.norm(p=2, dim=-1, keepdim=True)).cpu()


def precomputar_catalogo(lista_rutas, batch_size=BATCH_SIZE):
    catalogo = {}
    for i in tqdm(range(0, len(lista_rutas), batch_size), desc="Vectores"):
        batch = lista_rutas[i:i+batch_size]
        imagenes, ids = [], []
        for ruta in batch:
            try:
                imagenes.append(Image.open(ruta).convert("RGB"))
                ids.append(os.path.basename(ruta))
            except:
                pass
        if not imagenes:
            continue
        inputs = processor_clip(images=imagenes, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model_clip.get_image_features(**inputs)
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        for j, pid in enumerate(ids):
            catalogo[pid] = features.cpu()[j].unsqueeze(0)
    return catalogo


def precomputar_colores(lista_rutas):
    colores = {}
    for ruta in tqdm(lista_rutas, desc="Colores"):
        try:
            img = Image.open(ruta).convert("RGB")
            colores[os.path.basename(ruta)] = color_dominante(img)
        except:
            pass
    return colores


def buscar_top_k_con_color(recorte_pil, color_recorte, catalogo, colores_cat, k=TOP_K):
    if not catalogo:
        return []

    ids_todos = list(catalogo.keys())

    ids_filtrados = [
        pid for pid in ids_todos
        if pid in colores_cat and colores_similares(color_recorte, colores_cat[pid])
    ]

    if len(ids_filtrados) < 20:
        ids_filtrados = ids_todos

    query = obtener_embedding(recorte_pil).to(device)
    matriz = torch.cat([catalogo[pid] for pid in ids_filtrados], dim=0).to(device)
    similitudes = torch.matmul(query, matriz.T).squeeze()

    k_real = min(k, len(ids_filtrados))
    scores, idxs = torch.topk(similitudes, k_real)
    return [(ids_filtrados[idxs[i].item()], scores[i].item()) for i in range(k_real)]


rutas_catalogo = glob.glob(CARPETA_CATALOGO)
df_test = pd.read_csv(ARCHIVO_TEST)

if os.path.exists(ARCHIVO_VECTORES):
    diccionario_catalogo = torch.load(ARCHIVO_VECTORES)
else:
    diccionario_catalogo = precomputar_catalogo(rutas_catalogo)
    torch.save(diccionario_catalogo, ARCHIVO_VECTORES)

if os.path.exists(ARCHIVO_COLORES):
    colores_catalogo = torch.load(ARCHIVO_COLORES)
else:
    colores_catalogo = precomputar_colores(rutas_catalogo)
    torch.save(colores_catalogo, ARCHIVO_COLORES)

catalogos_por_categoria = {'top': {}, 'bottom': {}, 'shoe': {}, 'accesories': {}}
colores_por_categoria = {'top': {}, 'bottom': {}, 'shoe': {}, 'accesories': {}}

try:
    df_meta = pd.read_csv(ARCHIVO_METADATOS)
    df_meta['product_description'] = df_meta['product_description'].astype(str).str.lower()
    mapa = {}
    for _, row in df_meta.iterrows():
        desc = row['product_description']
        if any(p in desc for p in ['shirt', 'jacket', 'coat', 'dress', 'top', 'blouse', 'sweater', 't-shirt', 'blazer', 'outwear', 'vest']):
            cat = 'top'
        elif any(p in desc for p in ['trousers', 'jeans', 'skirt', 'shorts', 'pants', 'leggings']):
            cat = 'bottom'
        elif any(p in desc for p in ['shoe', 'sneaker', 'boot', 'sandal', 'heel', 'loafer', 'flats']):
            cat = 'shoe'
        else:
            cat = 'accesories'
        mapa[str(row['product_asset_id'])] = cat

    for pid_ext, vector in diccionario_catalogo.items():
        id_limpio = pid_ext.replace('.jpg', '').replace('.png', '')
        cat = mapa.get(id_limpio, 'accesories')
        catalogos_por_categoria[cat][pid_ext] = vector
        if pid_ext in colores_catalogo:
            colores_por_categoria[cat][pid_ext] = colores_catalogo[pid_ext]

except Exception:
    for cat in catalogos_por_categoria:
        catalogos_por_categoria[cat] = diccionario_catalogo
        colores_por_categoria[cat] = colores_catalogo

filas = []
vistos = set()

for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Bundles"):
    bundle_id = row['bundle_asset_id']
    ruta = os.path.join(CARPETA_BUNDLES, f"{bundle_id}.jpg")

    if not os.path.exists(ruta):
        continue

    try:
        for recorte in obtener_recortes_yolo(ruta):
            cat = recorte["categoria"]
            catalogo = catalogos_por_categoria.get(cat, diccionario_catalogo)
            colores_cat = colores_por_categoria.get(cat, colores_catalogo)

            for pid, score in buscar_top_k_con_color(recorte["imagen"], recorte["color"], catalogo, colores_cat):
                if score > UMBRAL:
                    id_limpio = pid.replace('.jpg', '').replace('.png', '')
                    clave = f"{bundle_id}_{id_limpio}"
                    if clave not in vistos:
                        vistos.add(clave)
                        filas.append({'bundle_asset_id': bundle_id, 'product_asset_id': id_limpio})
    except Exception:
        pass

df_entrega = pd.DataFrame(filas).drop_duplicates()
df_entrega.to_csv(ARCHIVO_SALIDA, index=False)
print(f"{len(df_entrega)} predicciones guardadas en {ARCHIVO_SALIDA}")
print(df_entrega.head(10))
