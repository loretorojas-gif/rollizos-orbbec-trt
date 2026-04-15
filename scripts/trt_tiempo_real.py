"""
trt_tiempo_real.py — Segmentacion y medicion de rollizos en tiempo real.

Pipeline:
  Orbbec Gemini 2 (RGB-D + D2C hardware)
      → TensorRT FP16 (RF-DETR-Seg)
          → Mascara por rollizo
              → Elipse minima → diametro real via formula pinhole

Arquitectura de hilos:
  FrameGrabber (hilo daemon) → siempre sobreescribe con el frame mas reciente
  Hilo principal             → toma el ultimo frame, infiere y muestra
  Sin buffer lag: la imagen se ve siempre en tiempo real.

Uso:
  python3 scripts/trt_tiempo_real.py
  DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority python3 scripts/trt_tiempo_real.py
"""
import os, sys, time, threading
import cv2
import numpy as np
import torch
import tensorrt as trt

# Agregar raiz del proyecto al path para importar config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg


# ── Hilo de captura ────────────────────────────────────────────────────────────

class FrameGrabber:
    """
    Hilo daemon que captura frames continuamente y mantiene solo el mas reciente.
    El hilo principal siempre lee el frame mas nuevo sin importar cuanto tarde
    la inferencia, eliminando el lag acumulativo del buffer del SDK.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.latest   = None          # tuple (bgr, depth_mm) o None
        self.lock     = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                continue
            cf = frames.get_color_frame()
            df = frames.get_depth_frame()
            if cf is None or df is None:
                continue

            color_data = np.frombuffer(cf.get_data(), dtype=np.uint8)
            bgr = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
            if bgr is None:
                continue

            depth_data = np.frombuffer(df.get_data(), dtype=np.uint16)
            scale      = df.get_depth_scale()   # Gemini 2: 1.0 (raw uint16 ya en mm)
            depth_mm   = depth_data.reshape(
                df.get_height(), df.get_width()
            ).astype(np.float32) * scale

            with self.lock:
                self.latest = (bgr, depth_mm)

    def get(self):
        """Devuelve el par (bgr, depth_mm) mas reciente, o None si no hay aun."""
        with self.lock:
            return self.latest

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)


# ── Motor TensorRT ─────────────────────────────────────────────────────────────

def cargar_engine():
    engine_path = os.path.join(os.path.dirname(__file__), '..', cfg.ENGINE_PATH)
    if not os.path.exists(engine_path):
        print(f"[ERROR] Motor no encontrado: {engine_path}")
        print("  Ejecuta primero:")
        print("    python3 setup/1_exportar_onnx.py")
        print("    python3 setup/2_compilar_trt.py")
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
    for n, t in tensors.items():
        if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
            print(f"  salida {n}: {tuple(t.shape)}")

    return engine, context, tensors


# ── Inferencia ─────────────────────────────────────────────────────────────────

def inferir(bgr, context, tensors, stream):
    """Normaliza, envia al motor TRT y retorna salidas como numpy."""
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
    """
    Calcula el diametro real de un rollizo usando su mascara y el mapa de profundidad.

    Formula pinhole:
        diametro_mm = (eje_menor_px * profundidad_mm) / FX

    Usa el eje menor de la elipse ajustada al contorno para ser robusto
    frente a rotaciones y perspectiva parcial del tronco.
    """
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
    try:
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
    except ImportError:
        print("[ERROR] pyorbbecsdk no instalado. Consulta README.md -> Instalacion.")
        sys.exit(1)

    engine, context, tensors = cargar_engine()
    stream = torch.cuda.current_stream()

    # Warm-up: elimina la latencia de la primera inferencia real
    print("[*] Warm-up TRT (3x dummy)...")
    dummy = np.zeros((cfg.COLOR_HEIGHT, cfg.COLOR_WIDTH, 3), dtype=np.uint8)
    for _ in range(3):
        inferir(dummy, context, tensors, stream)
    print("[OK] Warm-up completado.")

    # Configurar camara con alineacion D2C hardware
    print("[*] Iniciando camara Orbbec...")
    pipeline = Pipeline()
    config   = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile  = color_profiles.get_video_stream_profile(
        cfg.COLOR_WIDTH, cfg.COLOR_HEIGHT, OBFormat.MJPG, cfg.COLOR_FPS
    )
    config.enable_stream(color_profile)

    d2c_profiles = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
    if not d2c_profiles:
        print("[ERROR] La camara no soporta D2C hardware para esta resolucion.")
        print("  Verifica firmware o ajusta COLOR_WIDTH/COLOR_HEIGHT en config.py")
        sys.exit(1)
    config.enable_stream(d2c_profiles[0])
    config.set_align_mode(OBAlignMode.HW_MODE)

    pipeline.start(config)
    print("[OK] Camara iniciada — alineacion D2C hardware activa.")

    # Hilo de captura: siempre mantiene el frame mas reciente sin lag
    grabber = FrameGrabber(pipeline)
    print("[OK] Hilo de captura iniciado (sin buffer lag).")

    if cfg.GUARDAR_AUTO:
        os.makedirs(cfg.CARPETA_RESULTADOS, exist_ok=True)

    print()
    print("=" * 52)
    print("  SEGMENTACION TRT EN TIEMPO REAL")
    print("  Presiona  q  para salir")
    print("=" * 52)

    ultimo_guardado = 0.0
    total_guardados = 0
    fps_t0          = time.perf_counter()
    fps_count       = 0
    fps_display     = 0.0

    # Esperar el primer frame
    while grabber.get() is None:
        time.sleep(0.01)

    try:
        while True:
            frame = grabber.get()
            if frame is None:
                continue

            bgr, depth_mm = frame
            bgr = bgr.copy()   # copia local para no modificar el buffer compartido

            # Inferencia sobre el frame mas reciente
            t_inf   = time.perf_counter()
            salidas = inferir(bgr, context, tensors, stream)
            ms_inf  = (time.perf_counter() - t_inf) * 1000

            cajas    = salidas["dets"][0]       # (100, 4) cx,cy,w,h norm.
            logits   = salidas["labels"][0]     # (100, 91) logits de clase
            mascaras = salidas["masks"][0]      # (100, 96, 96) logits de mascara
            scores   = sigmoide(logits).max(axis=1)

            orig_h, orig_w = bgr.shape[:2]
            n_det = 0

            for i in range(100):
                if scores[i] < cfg.TRT_THRESHOLD:
                    continue
                # Mascara: sigmoid → resize a resolucion original → binarizar
                msk_full = cv2.resize(sigmoide(mascaras[i]),
                                      (orig_w, orig_h)) > 0.5
                med = medir_diametro(msk_full, depth_mm)
                if med is None:
                    continue
                n_det += 1
                dibujar_deteccion(bgr, msk_full, med, scores[i],
                                  cajas[i], orig_w, orig_h)

            # FPS del bucle principal
            fps_count += 1
            if fps_count >= 20:
                fps_display = fps_count / (time.perf_counter() - fps_t0)
                fps_t0      = time.perf_counter()
                fps_count   = 0

            # HUD superior
            hud = f"TRT  {ms_inf:.0f}ms | {fps_display:.1f} FPS | detecciones: {n_det}"
            cv2.putText(bgr, hud, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

            # Guardado automatico al detectar rollizos
            if cfg.GUARDAR_AUTO and n_det > 0:
                t_now = time.time()
                if t_now - ultimo_guardado >= cfg.INTERVALO_GUARDADO:
                    from datetime import datetime
                    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(cfg.CARPETA_RESULTADOS, f"trt_{ts}.jpg")
                    cv2.imwrite(path, bgr)
                    total_guardados += 1
                    ultimo_guardado  = t_now
                    print(f"  [SAVE] {path}")

            cv2.imshow("Rollizos TRT — Tiempo Real  (q = salir)", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nDetenido por el usuario.")
    finally:
        grabber.stop()
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[OK] Camara cerrada. Imagenes guardadas: {total_guardados}")


if __name__ == "__main__":
    main()
