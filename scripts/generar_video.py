"""
generar_video.py — Genera un video MP4 con detecciones sobre imágenes guardadas.

Lee pares (rgb, depth) de una carpeta, aplica el modelo TRT y produce un video
con máscaras, elipses, diámetros reales y HUD superpuesto.

Uso:
  python3 scripts/generar_video.py
  python3 scripts/generar_video.py --carpeta fotos_rollizo --fps 5 --salida demo.mp4
"""

import os, sys, glob, re, argparse
import cv2
import numpy as np
import torch
import tensorrt as trt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg


# ── Motor TensorRT ─────────────────────────────────────────────────────────────

def cargar_engine():
    engine_path = os.path.join(os.path.dirname(__file__), "..", cfg.ENGINE_PATH)
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


# ── Medición de diámetro ───────────────────────────────────────────────────────

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


# ── Visualización ──────────────────────────────────────────────────────────────

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


# ── Carga de pares rgb/depth ───────────────────────────────────────────────────

def cargar_pares(carpeta):
    """Empareja rgb_NNNN_*.jpg con depth_NNNN_*.png por número de frame."""
    rgbs = sorted(glob.glob(os.path.join(carpeta, "rgb_*.jpg")))
    pares = []
    for rgb_path in rgbs:
        nombre = os.path.basename(rgb_path)
        m = re.match(r"rgb_(\d+)_", nombre)
        if not m:
            continue
        n = m.group(1)
        depth_candidates = glob.glob(os.path.join(carpeta, f"depth_{n}_*.png"))
        if not depth_candidates:
            print(f"  [WARN] Sin depth para: {nombre}, saltando.")
            continue
        pares.append((rgb_path, depth_candidates[0]))
    return pares


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genera video MP4 con detecciones TRT")
    parser.add_argument("--carpeta", default="fotos_rollizo",
                        help="Carpeta con imágenes rgb_* y depth_* (default: fotos_rollizo)")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="FPS del video de salida (default: 5)")
    parser.add_argument("--salida", default="demo_rollizos.mp4",
                        help="Nombre del archivo MP4 de salida (default: demo_rollizos.mp4)")
    args = parser.parse_args()

    engine, context, tensors = cargar_engine()
    stream = torch.cuda.current_stream()

    print("[*] Warm-up TRT (3x dummy)...")
    dummy = np.zeros((cfg.COLOR_HEIGHT, cfg.COLOR_WIDTH, 3), dtype=np.uint8)
    for _ in range(3):
        inferir(dummy, context, tensors, stream)
    print("[OK] Warm-up completado.")

    pares = cargar_pares(args.carpeta)
    if not pares:
        print(f"[ERROR] No se encontraron pares rgb/depth en: {args.carpeta}")
        sys.exit(1)
    print(f"[*] {len(pares)} frames a procesar → {args.salida} @ {args.fps} FPS")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.salida, fourcc, args.fps,
                             (cfg.COLOR_WIDTH, cfg.COLOR_HEIGHT))
    if not writer.isOpened():
        print("[ERROR] No se pudo crear el archivo de video.")
        sys.exit(1)

    total_det = 0

    for idx, (rgb_path, depth_path) in enumerate(pares):
        bgr      = cv2.imread(rgb_path)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16, valores en mm

        if bgr is None or depth_raw is None:
            print(f"  [WARN] No se pudo leer: {os.path.basename(rgb_path)}, saltando.")
            continue

        depth_mm = depth_raw.astype(np.float32)  # Gemini 2: raw uint16 ya en mm

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

        total_det += n_det

        hud = f"Frame {idx + 1}/{len(pares)}  |  Rollizos detectados: {n_det}"
        cv2.putText(bgr, hud, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

        writer.write(bgr)
        print(f"  [{idx + 1:>3}/{len(pares)}] {os.path.basename(rgb_path)}  →  {n_det} det.")

    writer.release()
    print()
    print(f"[OK] Video guardado: {args.salida}")
    print(f"     Frames: {len(pares)}  |  Detecciones totales: {total_det}")
    print(f"     Duración aprox: {len(pares) / args.fps:.1f}s @ {args.fps} FPS")


if __name__ == "__main__":
    main()
