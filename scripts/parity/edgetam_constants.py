"""Writes the three learned tensors the EdgeTAM ONNX export leaves to the host into
data/edgetam_constants.json, from yonigozlan/EdgeTAM-hf at a pinned revision.

    python -I scripts/parity/edgetam_constants.py
"""
import json
import os

from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO, REVISION = "yonigozlan/EdgeTAM-hf", "c266ce53b3fc"
path = hf_hub_download(REPO, "model.safetensors", revision=REVISION)
with safe_open(path, "pt") as f:
    keys = list(f.keys())
    get = lambda name: f.get_tensor(name).float().reshape(-1 if name != TPOS else (7, 64)).tolist()
    TPOS = "memory_temporal_positional_encoding"
    occlusion = [k for k in keys if "occlusion_spatial_embedding" in k]
    out = {
        "source": f"https://huggingface.co/{REPO}/blob/{REVISION}/model.safetensors",
        "memory_temporal_positional_encoding": get(TPOS),
        "no_memory_embedding": get("no_memory_embedding"),
        "occlusion_spatial_embedding": get(occlusion[0]) if occlusion else None,
    }
print({k: (len(v) if isinstance(v, list) else v) for k, v in out.items()}, "occlusion keys:", occlusion)
dest = os.path.join(os.path.dirname(__file__), "..", "..", "data", "edgetam_constants.json")
os.makedirs(os.path.dirname(dest), exist_ok=True)
with open(dest, "w") as f:
    json.dump(out, f)
