"""
2_compilar_trt.py — Compila el ONNX exportado a un motor TensorRT FP16.

Requiere haber ejecutado primero:
  python3 setup/1_exportar_onnx.py

Tiempo estimado: 10–20 minutos (solo se hace una vez por modelo y dispositivo).
El motor compilado es especifico para la GPU de la Jetson donde se compila.

Uso:
  python3 setup/2_compilar_trt.py

Salida:
  trt_rollizos/rfdetr_rollizos_fp16.engine  (~64 MB)
"""
import os, sys
import tensorrt as trt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg

RUTA_ONNX   = os.path.join(os.path.dirname(__file__), '..', cfg.ONNX_DIR, 'inference_model.onnx')
RUTA_ENGINE = os.path.join(os.path.dirname(__file__), '..', cfg.ENGINE_PATH)

os.makedirs(os.path.dirname(RUTA_ENGINE), exist_ok=True)

if not os.path.exists(RUTA_ONNX):
    print(f"[ERROR] ONNX no encontrado: {RUTA_ONNX}")
    print("  Ejecuta primero: python3 setup/1_exportar_onnx.py")
    sys.exit(1)

print(f"[*] Compilando motor TRT desde: {RUTA_ONNX}")
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
builder    = trt.Builder(TRT_LOGGER)
config_trt = builder.create_builder_config()

# Workspace de 4 GB (necesario para compilar RF-DETR)
config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 ** 3)

# FP16: activo si la GPU lo soporta (Jetson Orin NX/AGX si)
if builder.platform_has_fast_fp16:
    config_trt.set_flag(trt.BuilderFlag.FP16)
    print("[*] Modo FP16 activado.")
else:
    print("[!] FP16 no disponible, compilando en FP32.")

config_trt.set_flag(trt.BuilderFlag.GPU_FALLBACK)

network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(RUTA_ONNX, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print("[ERR]", parser.get_error(i))
        raise RuntimeError("Fallo el parseo del ONNX.")

print("[*] Grafo ONNX parseado. Compilando (10-20 min)...")
motor = builder.build_serialized_network(network, config_trt)
if motor is None:
    raise RuntimeError("build_serialized_network devolvio None.")

with open(RUTA_ENGINE, "wb") as f:
    f.write(motor)

size_mb = os.path.getsize(RUTA_ENGINE) / 1024 / 1024
print(f"[OK] Motor guardado: {RUTA_ENGINE}  ({size_mb:.0f} MB)")
print()
print("Siguiente paso: python3 scripts/trt_tiempo_real.py")
