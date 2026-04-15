"""
1_exportar_onnx.py — Exporta el checkpoint RF-DETR-Seg a formato ONNX estatico.

Por que este paso:
  RF-DETR usa MultiScaleDeformableAttention, cuya implementacion base genera
  nodos ScatterND en el grafo ONNX, incompatibles con TensorRT. Este script
  aplica un parche que reemplaza la atencion deformable con una version estatica
  (spatial shapes fijos para resolucion 384x384) antes de la exportacion.

Uso:
  python3 setup/1_exportar_onnx.py

Salida:
  trt_rollizos/static/inference_model.onnx  (~125 MB)
"""
import os, warnings, sys
import torch
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as cfg

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', cfg.ONNX_DIR)
CHECKPOINT  = os.path.join(os.path.dirname(__file__), '..', cfg.CHECKPOINT_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Spatial shapes fijas para input 384x384 (n_levels=1, escala 32x32 = 1024 tokens)
SPATIAL_SHAPES = [[32, 32]]
SPLIT_SIZES    = [1024]


# ── Parche: atencion deformable con shapes estaticos ──────────────────────────
from rfdetr.utilities.tensors import _bilinear_grid_sample

def ms_deform_attn_core_pytorch_static(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """
    Version estatica de MultiScaleDeformableAttention.
    Elimina ScatterND del grafo ONNX al hardcodear las spatial shapes
    para resolucion 384x384 → compatible con TensorRT.
    """
    B, n_heads, head_dim, _ = value.shape
    _, Len_q, _, L, P, _   = sampling_locations.shape
    value_list      = value.split(SPLIT_SIZES, dim=3)
    sampling_grids  = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H, W) in enumerate(SPATIAL_SHAPES):
        value_l_         = value_list[lid_].view(B * n_heads, head_dim, H, W)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = _bilinear_grid_sample(
            value_l_, sampling_grid_l_, padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights_t = attention_weights.transpose(1, 2).reshape(
        B * n_heads, 1, Len_q, L * P
    )
    out    = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (out * attention_weights_t).sum(-1).view(B, n_heads * head_dim, Len_q)
    return output.transpose(1, 2).contiguous()


import rfdetr.models.ops.modules.ms_deform_attn as _attn_mod
_attn_mod.ms_deform_attn_core_pytorch = ms_deform_attn_core_pytorch_static
print("[OK] Parche de atencion deformable aplicado.")


# ── Cargar modelo ─────────────────────────────────────────────────────────────
from rfdetr import RFDETRSegSmall

if not os.path.exists(CHECKPOINT):
    print(f"[ERROR] Checkpoint no encontrado: {CHECKPOINT}")
    print("  Descarga el modelo entrenado y colócalo en la raiz del proyecto.")
    sys.exit(1)

print(f"[*] Cargando modelo desde: {CHECKPOINT}")
modelo = RFDETRSegSmall(pretrain_weights=CHECKPOINT)

from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn
count = sum(1 for _ in modelo.model.model.modules() if isinstance(_, MSDeformAttn))
print(f"    Modulos MSDeformAttn parcheados: {count}")


# ── Exportar ONNX ─────────────────────────────────────────────────────────────
print("[*] Exportando a ONNX (opset 17, shapes estaticos)...")
modelo.export(
    output_dir=OUTPUT_DIR,
    simplify=False,
    opset_version=17,
    verbose=False,
)

import glob
onnx_files = glob.glob(os.path.join(OUTPUT_DIR, "*.onnx"))
if not onnx_files:
    print("[ERROR] No se genero ningun archivo .onnx")
    sys.exit(1)

print(f"[OK] ONNX exportado: {onnx_files[0]}")


# ── Verificacion ──────────────────────────────────────────────────────────────
import onnx
m = onnx.load(onnx_files[0])
onnx.checker.check_model(m)
scatter_nodes = [n for n in m.graph.node if 'scatter' in n.op_type.lower()]
op_types = sorted(set(n.op_type for n in m.graph.node))
print(f"    Nodos ScatterND restantes: {len(scatter_nodes)}  (debe ser 0)")
print(f"    Op types totales: {len(op_types)}")
print()
print("Siguiente paso: python3 setup/2_compilar_trt.py")
