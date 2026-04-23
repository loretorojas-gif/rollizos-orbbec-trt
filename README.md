# Medición de rollizos en tiempo real

Segmentación y medición de diámetro de rollizos (troncos) en tiempo real usando
cámara RGB-D Orbbec Gemini 2 y motor TensorRT FP16 en NVIDIA Jetson.

---

## Requisitos

- NVIDIA Jetson con JetPack 6.x (torch y tensorrt incluidos)
- Cámara Orbbec Gemini 2 conectada por USB
- Python 3.10

```bash
sudo apt install python3-opencv
pip install rfdetr onnx pyorbbecsdk2
```

---

## Caso A — Misma Jetson (o mismo modelo de Jetson)

Te pasan el archivo `rfdetr_rollizos_fp16.engine` ya compilado.

```bash
git clone https://github.com/loretorojas-gif/rollizos-orbbec-trt.git
cd rollizos-orbbec-trt
mkdir -p trt_rollizos
cp /ruta/al/rfdetr_rollizos_fp16.engine trt_rollizos/
python3 scripts/trt_tiempo_real.py
```

---

## Caso B — Jetson diferente

Necesitas compilar el motor para tu GPU. Te pasan el archivo
`checkpoint_best_total.pth` o el `inference_model.onnx`.

```bash
git clone https://github.com/loretorojas-gif/rollizos-orbbec-trt.git
cd rollizos-orbbec-trt
```

**Si tienes el `.pth`**, exporta a ONNX primero (~5 min):

```bash
cp /ruta/al/checkpoint_best_total.pth .
python3 setup/1_exportar_onnx.py
```

**Si tienes el `.onnx`** directamente, cópialo:

```bash
mkdir -p trt_rollizos/static
cp /ruta/al/inference_model.onnx trt_rollizos/static/
```

Luego compila el motor para tu Jetson (~15 min, solo una vez):

```bash
python3 setup/2_compilar_trt.py
```

Ejecutar:

```bash
python3 scripts/trt_tiempo_real.py
```

---

## Ejecutar en tiempo real

```bash
# Desde la terminal de escritorio de la Jetson:
python3 scripts/trt_tiempo_real.py

# Desde SSH (con sesión gráfica activa en el monitor):
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority python3 scripts/trt_tiempo_real.py
```

Presiona `q` para salir.

---

## Inferencia sobre imágenes guardadas

Para procesar imágenes capturadas previamente (sin cámara en vivo):

```bash
cd ~/rollizos-orbbec-trt
python3 scripts/inferir_imagenes.py --carpeta <ruta_sesion> --guardar
```

**Parámetros:**

| Parámetro | Descripción |
|-----------|-------------|
| `--carpeta` | Ruta a la carpeta con imágenes RGB y Depth |
| `--guardar` | (Opcional) Guarda resultados en `resultados_inferencia/` |

**Ejemplo:**

```bash
python3 scripts/inferir_imagenes.py --carpeta ~/imagenes_test/2026-04-23T10-00-06_845766 --guardar
```

**Formato de imágenes esperado:**

```
sesion_XXXX/
├── rgb_0001_20260423_100015_123.jpg
├── depth_0001_20260423_100015_123.png   # PNG 16-bit (mm)
├── rgb_0002_20260423_100021_456.jpg
├── depth_0002_20260423_100021_456.png
└── ...
```

**Controles:**

- `SPACE` — Avanzar a la siguiente imagen
- `q` — Salir

Los resultados con las detecciones dibujadas quedan en `resultados_inferencia/`.

---

## Configuración (config.py)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `FX` | 689.51 | Focal length @ 1280×720. Cambiar si usas otra resolución o cámara. |
| `TRT_INPUT_RES` | 384 | Resolución de entrada del modelo. No cambiar. |
| `TRT_THRESHOLD` | 0.5 | Confianza mínima para aceptar una detección. |
| `GUARDAR_AUTO` | True | Guarda automáticamente en `resultados_trt_rt/` al detectar rollizos. |
| `INTERVALO_GUARDADO` | 5 | Segundos mínimos entre guardados consecutivos. |

---

## Estructura

```
rollizos-orbbec-trt/
├── config.py                  # Parámetros globales
├── scripts/
│   ├── trt_tiempo_real.py     # Inferencia en tiempo real con cámara
│   └── inferir_imagenes.py    # Inferencia sobre imágenes guardadas
└── setup/
    ├── 1_exportar_onnx.py     # .pth → .onnx (solo si Jetson diferente)
    └── 2_compilar_trt.py      # .onnx → .engine (solo si Jetson diferente)
```

---

## Archivos del modelo

Los archivos no están en el repo por su tamaño. Están disponibles en Google Drive:

**[Carpeta Google Drive — rollizos-orbbec-trt_modelos](https://drive.google.com/drive/folders/1j5OwLzNoitUCsEBrqYvY-v7hR3s_fqEz?usp=drive_link)**

| Archivo | Tamaño | Para qué sirve |
|---------|--------|----------------|
| `rfdetr_rollizos_fp16.engine` | 64 MB | Motor TRT listo para usar (Caso A) |
| `inference_model.onnx` | 118 MB | Para compilar en otra Jetson (Caso B) |
| `checkpoint_best_total.pth` | 129 MB | Pesos originales del modelo (Caso B) |
