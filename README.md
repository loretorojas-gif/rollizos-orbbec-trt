# Medicion de rollizos en tiempo real

Segmentacion y medicion de diametro de rollizos (troncos) en tiempo real usando
camara RGB-D Orbbec Gemini 2 y motor TensorRT FP16 en NVIDIA Jetson.

---

## Requisitos

- NVIDIA Jetson con JetPack 6.x (torch y tensorrt incluidos)
- Camara Orbbec Gemini 2 conectada por USB
- Python 3.10

```bash
sudo apt install python3-opencv
pip install rfdetr onnx pyorbbecsdk2
```

---

## Caso A — Misma Jetson (o mismo modelo de Jetson)

Te pasan el archivo `rfdetr_rollizos_fp16.engine` ya compilado.

```bash
git clone https://github.com/tu-usuario/rollizos-orbbec-trt.git
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
git clone https://github.com/tu-usuario/rollizos-orbbec-trt.git
cd rollizos-orbbec-trt
```

**Si tienes el `.pth`**, exporta a ONNX primero (~5 min):
```bash
cp /ruta/al/checkpoint_best_total.pth .
python3 setup/1_exportar_onnx.py
```

**Si tienes el `.onnx`** directamente, copialo:
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

## Ejecutar

```bash
# Desde la terminal de escritorio de la Jetson:
python3 scripts/trt_tiempo_real.py

# Desde SSH (con sesion grafica activa en el monitor):
DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority python3 scripts/trt_tiempo_real.py
```

Presiona `q` para salir.

---

## Configuracion (config.py)

| Parametro | Valor | Descripcion |
|---|---|---|
| `FX` | 689.51 | Focal length @ 1280×720. Cambiar si usas otra resolucion o camara. |
| `TRT_INPUT_RES` | 384 | Resolucion de entrada del modelo. No cambiar. |
| `TRT_THRESHOLD` | 0.5 | Confianza minima para aceptar una deteccion. |
| `GUARDAR_AUTO` | True | Guarda automaticamente en `resultados_trt_rt/` al detectar rollizos. |
| `INTERVALO_GUARDADO` | 5 | Segundos minimos entre guardados consecutivos. |

---

## Estructura

```
rollizos-orbbec-trt/
├── config.py                  # Parametros globales
├── scripts/
│   └── trt_tiempo_real.py     # Script principal
└── setup/
    ├── 1_exportar_onnx.py     # .pth → .onnx  (solo si Jetson diferente)
    └── 2_compilar_trt.py      # .onnx → .engine (solo si Jetson diferente)
```

## Archivos del modelo

Los archivos no estan en el repo por su tamaño. Estan disponibles en Google Drive:

**[Carpeta Google Drive — rollizos-orbbec-trt_modelos](https://drive.google.com/drive/folders/1j5OwLzNoitUCsEBrqYvY-v7hR3s_fqEz?usp=drive_link)**

| Archivo | Tamaño | Para que sirve |
|---|---|---|
| `rfdetr_rollizos_fp16.engine` | 64 MB | Motor TRT listo para usar (Caso A) |
| `inference_model.onnx` | 118 MB | Para compilar en otra Jetson (Caso B) |
| `checkpoint_best_total.pth` | 129 MB | Pesos originales del modelo (Caso B) |

> Reemplaza el link con el de tu carpeta de Drive una vez que la compartas.
