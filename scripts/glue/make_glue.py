"""Builds the small ONNX graphs that keep EdgeTAM's per-frame work on the GPU between the model's own graphs:
data/glue/*.onnx. Each is a few ops; the constants come from edgetam_constants.json.

    python -I scripts/glue/make_glue.py      (needs the onnx package)
"""
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OPSET = [helper.make_opsetid("", 18)]
F, I64, U8, B = TensorProto.FLOAT, TensorProto.INT64, TensorProto.UINT8, TensorProto.BOOL


def const(name, array):
    return numpy_helper.from_array(np.asarray(array), name)


def save(name, nodes, inputs, outputs, initializers=()):
    graph = helper.make_graph(nodes, name, inputs, outputs, list(initializers))
    model = helper.make_model(graph, opset_imports=OPSET, producer_name="edgetam-onnx")
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, os.path.join(ROOT, "glue", f"{name}.onnx"))


def main():
    constants = json.load(open(os.path.join(ROOT, "edgetam_constants.json")))
    os.makedirs(os.path.join(ROOT, "glue"), exist_ok=True)
    io = helper.make_tensor_value_info

    # RGB frame as decoded (H, W, 3 bytes) -> the encoder's normalised 1024² NCHW input. Antialiased bilinear, as
    # transformers' fast SAM 2 processor resizes (torchvision, antialias=True).
    save("preprocess", [
        helper.make_node("Cast", ["rgb"], ["f"], to=F),
        helper.make_node("Transpose", ["f"], ["chw"], perm=[2, 0, 1]),
        helper.make_node("Unsqueeze", ["chw", "axis0"], ["nchw"]),
        helper.make_node("Resize", ["nchw", "", "", "size"], ["resized"], mode="linear", antialias=1,
                         coordinate_transformation_mode="half_pixel"),
        helper.make_node("Mul", ["resized", "scale"], ["unit"]),
        helper.make_node("Sub", ["unit", "mean"], ["centred"]),
        helper.make_node("Div", ["centred", "std"], ["pixel_values"]),
    ], [io("rgb", U8, ["height", "width", 3])], [io("pixel_values", F, [1, 3, 1024, 1024])], [
        const("axis0", np.array([0], np.int64)),
        const("size", np.array([1, 3, 1024, 1024], np.int64)),
        const("scale", np.array(1 / 255, np.float32)),
        const("mean", np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)),
        const("std", np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)),
    ])

    # [1, 256, 64, 64] -> [4096, 1, 256], the memory attention's sequence-first layout.
    save("to_sequence", [
        helper.make_node("Reshape", ["map", "flat"], ["chw"]),
        helper.make_node("Transpose", ["chw"], ["sequence"], perm=[2, 0, 1]),
    ], [io("map", F, [1, 256, 64, 64])], [io("sequence", F, [4096, 1, 256])],
        [const("flat", np.array([1, 256, 4096], np.int64))])

    # The memory attention's output (4096 tokens of 256, any leading layout) -> [1, 256, 64, 64].
    save("to_map", [
        helper.make_node("Reshape", ["tokens", "rows"], ["flat"]),
        helper.make_node("Transpose", ["flat"], ["chw"], perm=[1, 0]),
        helper.make_node("Reshape", ["chw", "square"], ["map"]),
    ], [io("tokens", F, ["d0", "d1", "d2", 256])], [io("map", F, [1, 256, 64, 64])], [
        const("rows", np.array([4096, 256], np.int64)),
        const("square", np.array([1, 256, 64, 64], np.int64)),
    ])

    # The seed frame decodes on its features plus the learned no-memory embedding.
    no_memory = np.array(constants["no_memory_embedding"], np.float32).reshape(1, 256, 1, 1)
    save("add_no_memory", [helper.make_node("Add", ["map", "no_memory"], ["seed_map"])],
         [io("map", F, [1, 256, 64, 64])], [io("seed_map", F, [1, 256, 64, 64])], [const("no_memory", no_memory)])

    # Positions of n spatial memories: every memory shares the same 512 latent positions; each slot adds the temporal
    # row of its age (row 6 for the seed, k - 1 for a frame k back).
    temporal = np.array(constants["memory_temporal_positional_encoding"], np.float32)  # [7, 64]
    save("bank_positions", [
        helper.make_node("Gather", ["temporal", "rows"], ["per_slot"], axis=0),           # [n, 64]
        helper.make_node("Unsqueeze", ["per_slot", "axis1"], ["per_slot3"]),             # [n, 1, 64]
        helper.make_node("Reshape", ["latent_positions", "latents"], ["latents3"]),      # [1, 512, 64]
        helper.make_node("Add", ["latents3", "per_slot3"], ["summed"]),                 # [n, 512, 64]
        helper.make_node("Reshape", ["summed", "sequence"], ["positions"]),             # [n·512, 1, 64]
    ], [io("latent_positions", F, [1, 512, 64]), io("rows", I64, ["n"])], [io("positions", F, ["tokens", 1, 64])], [
        const("temporal", temporal),
        const("axis1", np.array([1], np.int64)),
        const("latents", np.array([1, 512, 64], np.int64)),
        const("sequence", np.array([-1, 1, 64], np.int64)),
    ])

    # Our decoder's four masks (scripts/export/edgetam_decoder.py) -> the one transformers keeps, at 1024², and its
    # pointer. multimask (one point or none): the best of tokens 1–3 by predicted IoU, mask and pointer. Otherwise
    # token 0's mask unless unstable (_dynamic_multimask_via_stability), and token 0's pointer either way. Absent
    # objects get NO_OBJ_SCORE logits, as in _single_frame_forward. Delta and threshold: the mask decoder config's.
    save("select_mask", [
        helper.make_node("Slice", ["iou_scores", "one", "four", "axis2"], ["candidates"]),
        helper.make_node("ArgMax", ["candidates"], ["best0"], axis=2, keepdims=0),          # [1, 1]
        helper.make_node("Reshape", ["best0", "flat1"], ["best1"]),
        helper.make_node("Add", ["best1", "one"], ["best"]),                                # [1], in 1–3
        helper.make_node("Gather", ["masks", "zero_i"], ["single"], axis=2),                # [1, 1, 256, 256]
        helper.make_node("Greater", ["single", "delta"], ["inner"]),
        helper.make_node("Greater", ["single", "minus_delta"], ["outer"]),
        helper.make_node("Cast", ["inner"], ["inner_f"], to=F),
        helper.make_node("Cast", ["outer"], ["outer_f"], to=F),
        helper.make_node("ReduceSum", ["inner_f"], ["area_i"], keepdims=0),
        helper.make_node("ReduceSum", ["outer_f"], ["area_u"], keepdims=0),
        helper.make_node("Max", ["area_u", "one_f"], ["area_u1"]),
        helper.make_node("Div", ["area_i", "area_u1"], ["ratio"]),
        helper.make_node("Greater", ["area_u", "zero_f"], ["any_u"]),
        helper.make_node("Where", ["any_u", "ratio", "one_f"], ["stability"]),
        helper.make_node("GreaterOrEqual", ["stability", "thresh"], ["stable"]),
        helper.make_node("Where", ["stable", "zero_1", "best"], ["single_index"]),
        helper.make_node("Where", ["multimask", "best", "single_index"], ["mask_index"]),
        helper.make_node("Where", ["multimask", "best", "zero_1"], ["pointer_index"]),
        helper.make_node("Gather", ["masks", "mask_index"], ["chosen5"], axis=2),
        helper.make_node("Reshape", ["chosen5", "low_res"], ["chosen"]),
        helper.make_node("Greater", ["object_score_logits", "zero_f"], ["appearing"]),     # [1, 1, 1]
        helper.make_node("Unsqueeze", ["appearing", "axis3"], ["appearing4"]),
        helper.make_node("Where", ["appearing4", "chosen", "no_object"], ["gated"]),
        helper.make_node("Resize", ["gated", "", "", "high_res"], ["high_res_masks"], mode="linear",
                         coordinate_transformation_mode="half_pixel"),
        helper.make_node("Gather", ["object_pointers", "pointer_index"], ["pointer4"], axis=2),
        helper.make_node("Reshape", ["pointer4", "pointer_shape"], ["object_pointer"]),
    ], [io("masks", F, [1, 1, 4, 256, 256]), io("iou_scores", F, [1, 1, 4]), io("object_pointers", F, [1, 1, 4, 256]),
        io("object_score_logits", F, [1, 1, 1]), io("multimask", B, [1])],
        [io("high_res_masks", F, [1, 1, 1024, 1024]), io("object_pointer", F, [1, 1, 256])], [
        const("one", np.array([1], np.int64)),
        const("four", np.array([4], np.int64)),
        const("axis2", np.array([2], np.int64)),
        const("axis3", np.array([3], np.int64)),
        const("flat1", np.array([1], np.int64)),
        const("zero_i", np.array(0, np.int64)),
        const("zero_1", np.array([0], np.int64)),
        const("zero_f", np.array(0, np.float32)),
        const("one_f", np.array(1, np.float32)),
        const("delta", np.array(0.05, np.float32)),
        const("minus_delta", np.array(-0.05, np.float32)),
        const("thresh", np.array(0.98, np.float32)),
        const("no_object", np.array(-1024, np.float32)),
        const("low_res", np.array([1, 1, 256, 256], np.int64)),
        const("high_res", np.array([1, 1, 1024, 1024], np.int64)),
        const("pointer_shape", np.array([1, 1, 256], np.int64)),
    ])

    # 1024² mask logits -> the object at the video's size, bilinear as transformers' post-processing, > 0.
    save("mask_out", [
        helper.make_node("Concat", ["batch", "size"], ["sizes"], axis=0),
        helper.make_node("Resize", ["logits", "", "", "sizes"], ["resized"], mode="linear",
                         coordinate_transformation_mode="half_pixel"),
        helper.make_node("Greater", ["resized", "zero"], ["above"]),
        helper.make_node("Cast", ["above"], ["mask"], to=U8),
    ], [io("logits", F, [1, 1, 1024, 1024]), io("size", I64, [2])], [io("mask", U8, [1, 1, "height", "width"])], [
        const("batch", np.array([1, 1], np.int64)),
        const("zero", np.array(0, np.float32)),
    ])
    print("glue graphs written to", os.path.abspath(os.path.join(ROOT, "glue")))


if __name__ == "__main__":
    main()
