"""
config.py — Configuracion centralizada del sistema de medicion de rollizos.
Edita este archivo para adaptar el sistema a tu camara o resolucion.
"""

# ── Rutas ──────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoint_best_total.pth"   # pesos PyTorch del modelo
ONNX_DIR        = "trt_rollizos/static"          # directorio de salida ONNX
ENGINE_PATH     = "trt_rollizos/rfdetr_rollizos_fp16.engine"  # motor TensorRT

# ── Camara Orbbec ──────────────────────────────────────────────────────────────
COLOR_WIDTH  = 1280
COLOR_HEIGHT = 720
COLOR_FPS    = 30
COLOR_FORMAT = "MJPG"        # formato del sensor color

# ── Parametros opticos (Orbbec Gemini 2 @ 1280x720 con D2C) ───────────────────
# Para otra camara u otra resolucion: obtener FX con get_camera_param().rgb_intrinsic.fx
FX = 689.51                  # focal length horizontal del sensor color [pixeles]

# Gemini 2: los valores uint16 del depth ya estan en mm (scale=1.0)
# Si tu camara devuelve scale distinto, la formula lo compensa automaticamente.

# ── Motor TensorRT ─────────────────────────────────────────────────────────────
TRT_INPUT_RES  = 384         # resolucion de entrada del modelo [pixeles cuadrados]
TRT_THRESHOLD  = 0.5         # umbral de confianza para aceptar detecciones

# ── Rendimiento en tiempo real ─────────────────────────────────────────────────
SKIP_FRAMES = 2              # ejecutar inferencia cada N frames capturados
                             # 1 = todos los frames (maximo precision, mas GPU)
                             # 2 = uno si uno no (recomendado, ~17 FPS netos)
                             # 3 = cada 3 frames (modo bajo consumo)

# ── Guardado automatico ────────────────────────────────────────────────────────
GUARDAR_AUTO       = True
CARPETA_RESULTADOS = "resultados_trt_rt"
INTERVALO_GUARDADO = 6       # segundos minimos entre guardados consecutivos

# ── Medicion ───────────────────────────────────────────────────────────────────
PROF_MAX_MM = 5000           # ignorar lecturas de profundidad > 5 m (ruido)
