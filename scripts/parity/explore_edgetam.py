"""One-off: compare the EdgeTAM ONNX graphs with transformers' EdgeTamVideoModel stage by stage on one frame."""
import sys
import numpy as np, torch, onnxruntime as ort
from PIL import Image
from transformers import EdgeTamVideoModel, Sam2VideoProcessor

root, models = sys.argv[1], sys.argv[2]
model = EdgeTamVideoModel.from_pretrained("yonigozlan/EdgeTAM-hf", revision="c266ce53b3fc").eval()
proc = Sam2VideoProcessor.from_pretrained("yonigozlan/EdgeTAM-hf", revision="c266ce53b3fc")
cfg = model.config
print("config:", {k: getattr(cfg, k, None) for k in ["multimask_output_in_sam", "multimask_output_for_tracking",
      "multimask_min_pt_num", "multimask_max_pt_num", "max_object_pointers_in_encoder",
      "enable_temporal_pos_encoding_for_object_pointers", "enable_occlusion_spatial_embedding",
      "sigmoid_scale_for_mem_enc", "sigmoid_bias_for_mem_enc", "max_cond_frame_num"]})
print("params:", [n for n, _ in model.named_parameters() if "." not in n])
print("tpos", tuple(model.memory_temporal_positional_encoding.shape), "no_mem", tuple(model.no_memory_embedding.shape),
      "occl", None if model.occlusion_spatial_embedding_parameter is None else tuple(model.occlusion_spatial_embedding_parameter.shape))
print("processor:", proc.image_processor)

img = Image.open(f"{root}/00000.jpg").convert("RGB")
pix = proc(images=img, return_tensors="pt")["pixel_values"]
print("pixel_values", tuple(pix.shape), float(pix.mean()), float(pix.std()))
s = lambda name: ort.InferenceSession(f"{models}/{name}.onnx", providers=["CPUExecutionProvider"])
enc = s("vision_encoder").run(None, {"pixel_values": pix.numpy()})
names = [o.name for o in s("vision_encoder").get_outputs()]
with torch.no_grad():
    feats, pos = model.get_image_features(pix)[:2] if isinstance(model.get_image_features(pix), tuple) else (None, None)
out = model.get_image_features(pix)
print("get_image_features returns", type(out), [type(x) for x in out] if isinstance(out, tuple) else None)
for n, a in zip(names, enc): print(" onnx", n, a.shape, float(a.mean()))
