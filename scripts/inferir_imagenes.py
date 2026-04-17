"""
inferir_imagenes.py — Segmentacion y medicion de rollizos sobre imagenes guardadas.

Uso:
  python3 scripts/inferir_imagenes.py --carpeta <ruta_imagenes> [--guardar]

  --carpeta   Carpeta con pares rgb_*/depth_* capturados con el script de captura
  --guardar   Guarda los resultados en resultados_inferencia/ (opcional)

Formato esperado (mismo timestamp):
  rgb_0007_20260417_092415_629.jpg   (o .png)
  depth_0007_20260417_092415_629.npy (o .png 16-bit)
"""
import os, sys, argparse, glob
import cv2
import numpy as np
import torch
import tensorrt as trt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import config as cfg


# ── Motor TensorRT ─────────────────────────────────────────────────────────────

def cargar_engine():
    engine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', cfg.ENGINE_PATH)
    if not os.path.exists(engine_path):
        print(f"[ERROR] Motor no encontrado: {engine_path}")
        sys.exit(1)

    print(f"[*] Cargando motor TRT: {engine_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    tensors = {}
    for i in range(engine.num_io_tensors):
        name     = engine.get_tensor_name(i)
        shape    = tuple(engine.get_tensor_shape(name))
        dtype_np = trt.nptype(engine.get_tensor_dtype(name))
        dtype_t  = torch.float16 if dtype_np == np.float16 else torch.float32
        t = torch.zeros([abs(d) for d in shape], dtype=dtype_t, device="cuda")
        context.set_tensor_address(name, t.data_ptr())
        tensors[name] = t

    print("[OK] Motor cargado.")
    return engine, context, tensors


# ── Inferencia ─────────────────────────────────────────────────────────────────

def inferir(bgr, context, tensors, stream):
    resized = cv2.resize(bgr, (cfg.TRT_INPUT_RES, cfg.TRT_INPUT_RES))
    inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensors["input"].copy_(torch.from_numpy(inp.transpose(2, 0, 1)[np.newaxis]).cuda())
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    torch.cuda.synchronize()
    return {n: tensors[n].float().cpu().numpy()
            for n in tensors
            if context.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT}


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


# ── Medicion de diametro ───────────────────────────────────────────────────────

def medir_diametro(mask_bool, depth_mm):
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None

    ellipse = cv2.fitEllipseDirect(contour)
    (_, _), (ma, MA), _ = ellipse
    eje_menor_px = min(ma, MA)

    vals = depth_mm[mask_bool]
    vals = vals[vals > 0]
    if vals.size == 0:
        return None

    prof_mm = float(np.median(vals))
    if prof_mm <= 0 or prof_mm > cfg.PROF_MAX_MM:
        return None

    diametro_cm = (eje_menor_px * prof_mm) / cfg.FX / 10.0
    return {
        "diametro_cm":   diametro_cm,
        "profundidad_m": prof_mm / 1000.0,
        "ellipse":       ellipse,
    }


# ── Visualizacion ──────────────────────────────────────────────────────────────

def dibujar_deteccion(bgr, msk_full, med, score, caja, orig_w, orig_h):
    overlay = bgr.copy()
    overlay[msk_full] = (0, 180, 80)
    cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0, bgr)

    contours, _ = cv2.findContours(msk_full.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, (0, 255, 100), 2)
    cv2.ellipse(bgr, med["ellipse"], (255, 100, 0), 2)

    cx_b, cy_b, bw_b, bh_b = caja
    x1 = int((cx_b - bw_b / 2) * orig_w)
    y1 = int((cy_b - bh_b / 2) * orig_h)
    label = f"D={med['diametro_cm']:.1f}cm  Z={med['profundidad_m']:.2f}m  {score:.0%}"
    cv2.rectangle(bgr, (x1, y1 - 26), (x1 + len(label) * 10, y1), (20, 20, 20), -1)
    cv2.putText(bgr, label, (x1 + 5, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carpeta", required=True, help="Carpeta con imagenes a procesar")
    parser.add_argument("--guardar", action="store_true", help="Guardar resultados en resultados_inferencia/")
    args = parser.parse_args()

    if not os.path.isdir(args.carpeta):
        print(f"[ERROR] Carpeta no encontrada: {args.carpeta}")
        sys.exit(1)

    imagenes = sorted(
        glob.glob(os.path.join(args.carpeta, "rgb_*.jpg")) +
        glob.glob(os.path.join(args.carpeta, "rgb_*.png"))
    )
    if not imagenes:
        print(f"[ERROR] No se encontraron archivos rgb_* en: {args.carpeta}")
        sys.exit(1)

    print(f"[*] {len(imagenes)} imagenes encontradas.")

    engine, context, tensors = cargar_engine()
    stream = torch.cuda.current_stream()

    print("[*] Warm-up TRT...")
    dummy = np.zeros((cfg.COLOR_HEIGHT, cfg.COLOR_WIDTH, 3), dtype=np.uint8)
    for _ in range(3):
        inferir(dummy, context, tensors, stream)
    print("[OK] Warm-up completado.\n")

    if args.guardar:
        out_dir = "resultados_inferencia"
        os.makedirs(out_dir, exist_ok=True)

    print("Presiona  SPACE  para avanzar,  q  para salir.\n")

    for img_path in imagenes:
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"[SKIP] No se pudo leer: {img_path}")
            continue

        # Buscar depth con mismo timestamp: rgb_XXXX → depth_XXXX
        nombre    = os.path.basename(img_path)
        timestamp = nombre[len("rgb_"):]                        # ej: 0007_20260417_092415_629.jpg
        base_ts   = os.path.splitext(timestamp)[0]             # ej: 0007_20260417_092415_629
        carpeta   = os.path.dirname(img_path)

        npy_path = os.path.join(carpeta, f"depth_{base_ts}.npy")
        png_path = os.path.join(carpeta, f"depth_{base_ts}.png")

        if os.path.exists(npy_path):
            depth_mm    = np.load(npy_path).astype(np.float32)
            tiene_depth = True
        elif os.path.exists(png_path):
            depth_mm    = cv2.imread(png_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            tiene_depth = True
        else:
            depth_mm    = np.zeros(bgr.shape[:2], dtype=np.float32)
            tiene_depth = False

        salidas  = inferir(bgr, context, tensors, stream)
        cajas    = salidas["dets"][0]
        logits   = salidas["labels"][0]
        mascaras = salidas["masks"][0]
        scores   = sigmoide(logits).max(axis=1)

        orig_h, orig_w = bgr.shape[:2]
        n_det = 0

        for i in range(100):
            if scores[i] < cfg.TRT_THRESHOLD:
                continue
            msk_full = cv2.resize(sigmoide(mascaras[i]), (orig_w, orig_h)) > 0.5
            med = medir_diametro(msk_full, depth_mm)
            if med is None:
                continue
            n_det += 1
            dibujar_deteccion(bgr, msk_full, med, scores[i], cajas[i], orig_w, orig_h)

        tag_depth = "" if tiene_depth else "  [sin depth]"
        hud = f"{os.path.basename(img_path)}  |  detecciones: {n_det}{tag_depth}"
        cv2.putText(bgr, hud, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

        print(f"  {os.path.basename(img_path):30s}  detecciones: {n_det}{tag_depth}")

        if args.guardar:
            out_path = os.path.join(out_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, bgr)

        cv2.imshow("Inferencia sobre imagenes  (SPACE=siguiente  q=salir)", bgr)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    print("\n[OK] Listo.")


if __name__ == "__main__":
    main()
