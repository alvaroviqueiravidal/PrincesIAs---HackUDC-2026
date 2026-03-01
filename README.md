
![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_Enabled-EE4C2C?style=flat-square&logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-00FFFF?style=flat-square)
![HuggingFace](https://img.shields.io/badge/Fashion--CLIP-Transformers-FFD21E?style=flat-square&logo=huggingface)
![OpenCV](https://img.shields.io/badge/OpenCV-Color_Analysis-5C3EE8?style=flat-square&logo=opencv)

## 🚀 Descripción del Proyecto

Este proyecto intenta resolver el desafío de identificar prendas individuales dentro de la fotografía de un look completo (bundle) y emparejarlas con el producto exacto dentro de un catálogo de más de 27.000 artículos. 

Para lograr la máxima precisión, hemos construido un **Pipeline Híbrido** acelerado por GPU que combina:
1. **Inteligencia Artificial Semántica** (para entender la forma, tejido y estilo de la prenda).
2. **Visión Computacional Clásica** (para extraer matemáticamente el color exacto y penalizar resultados con colores distintos).

## 🧠 Arquitectura del Sistema

Nuestro pipeline se divide en 5 fases altamente optimizadas:

1. **Detección y Recorte (YOLOv8):** Utilizamos un modelo de segmentación `deepfashion2_yolov8s-seg` para aislar dinámicamente cada prenda (pantalones, camisetas, zapatos) de la imagen original del modelo.
2. **Análisis de Color (K-Means):** Extraemos el color dominante del recorte ignorando el fondo (blanco/transparente) utilizando clustering con K-Means sobre el espacio RGB.
3. **Filtro Inteligente por Metadatos:** Ingestamos el `product_dataset.csv` de Inditex para mapear las descripciones en 4 macro-categorías (`top`, `bottom`, `shoe`, `accesories`). Esto enruta la búsqueda y elimina los falsos positivos estructurales.
4. **Embeddings de Dominio (Fashion-CLIP):** Usamos `patrickjohncyh/fashion-clip`, un modelo preentrenado en moda. Precomputamos tanto los tensores (vectores de 512D) como el color dominante de los 27k productos, almacenándolos en caché (`.pt`) para una carga ultrarrápida.
5. **Scoring Combinado (Similitud + Color):** Sustituimos la simple similitud matemática por una métrica avanzada. Calculamos la similitud coseno (PyTorch) y le restamos una **penalización de color** basada en la distancia CIELAB (Delta E CIE76). Extraemos el Top-15 global por *bundle* para maximizar la métrica de *Recall*.

---

## 📁 Estructura del Proyecto

```text
proyecto_inditex/
├── data/
│   ├── catalog_images/        # Catálogo de 27k+ productos
│   ├── raw_images/            # Imágenes de los bundles (modelos)
│   └── bundles_product_match_test.csv  # Dataset de evaluación
├── metadata/
│   └── product_dataset.csv    # Metadatos y descripciones de Inditex
├── main.py          # Script principal de inferencia (con análisis de color)
├── deepfashion2_yolov8s-seg.pt# Pesos del modelo YOLOv8
├── vectores_catalogo_27k.pt   # Caché de embeddings y colores (Autogenerado)
└── README.md                  # Este archivo
