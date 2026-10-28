"""Exports EdgeTAM's prompt encoder and mask decoder from transformers' EdgeTamVideoModel with every mask kept:
the single-mask token's and the three multimask candidates, so the caller picks as transformers does (best of three
for one point or none, single mask with the stability fallback for two or more). The ready-made export
(jax-image-tools/edgetam-video-onnx) always takes the best of three.

    python -I scripts/export/edgetam_decoder.py [out folder, default .models/edgetam-decoder]   (CUDA torch venv)
    python -I scripts/export/edgetam_decoder.py --patch <mask_decoder.onnx>   (only labels_as_float, in place)

Inputs as that export's mask_decoder.onnx: feats0 [1,32,256,256], feats1 [1,64,128,128], feats2 [1,256,64,64],
input_points [1,1,N,2] (1024² space), input_labels [1,1,N] int32. Outputs: masks [1,1,4,256,256] (logits, not
gated), iou_scores [1,1,4], object_pointers [1,1,4,256] (one per token, gated by the object score as in
_single_frame_forward), object_score_logits [1,1,1]. Index 0 is the single-mask token.

The labels' padding and clamping are done in float (labels_as_float): ORT has no CUDA kernels for int32 Pad and Clip,
and a graph with any node on the CPU cannot be recorded as a CUDA graph.
"""
import hashlib
import os
import sys

import numpy as np
import torch
from torch import nn

REPO, REVISION = "yonigozlan/EdgeTAM-hf", "c266ce53b3fc"


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("image_positions", model.get_image_wide_positional_embeddings().detach().clone())

    def forward(self, feats0, feats1, feats2, input_points, input_labels):
        model, dec = self.model, self.model.mask_decoder
        sparse, dense = model.prompt_encoder(
            input_points=input_points, input_labels=input_labels, input_boxes=None, input_masks=None)

        # EdgeTamVideoMaskDecoder.forward up to its selection, for batch 1 and one object.
        tokens = torch.cat([dec.obj_score_token.weight, dec.iou_token.weight, dec.mask_tokens.weight], dim=0)
        tokens = torch.cat((tokens.reshape(1, 1, -1, tokens.shape[-1]), sparse), dim=2)
        tokens, image = dec.transformer(
            point_embeddings=tokens, image_embeddings=feats2 + dense, image_positional_embeddings=self.image_positions,
            attention_similarity=None)
        mask_tokens = tokens[:, :, 2:2 + dec.num_mask_tokens, :]
        image = image.transpose(2, 3).reshape(1, 256, 64, 64)
        up = dec.upscale_conv1(image) + feats1
        up = dec.activation(dec.upscale_layer_norm(up))
        up = dec.activation(dec.upscale_conv2(up) + feats0)
        hyper = torch.stack([dec.output_hypernetworks_mlps[i](mask_tokens[:, :, i, :])
                             for i in range(dec.num_mask_tokens)], dim=2)
        masks = (hyper @ up.reshape(1, 1, up.shape[1], -1)).reshape(1, 1, -1, 256, 256)
        iou_scores = dec.iou_prediction_head(tokens[:, :, 1, :])
        object_score_logits = dec.pred_obj_score_head(tokens[:, :, 0, :])

        # The pointer of every token, gated as _single_frame_forward gates the chosen one.
        appearing = (object_score_logits > 0).to(feats2.dtype).unsqueeze(2)
        pointers = model.object_pointer_proj(mask_tokens)
        pointers = appearing * pointers + (1 - appearing) * model.no_object_pointer
        return masks, iou_scores, pointers, object_score_logits


def labels_as_float(path):
    """Pads and clamps the point labels in float and casts the result to the embedding's int64 index. Labels are
    small integers, so the values are exact."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(path)
    graph = model.graph
    nodes = {n.name: n for n in graph.node}
    pad, clip = nodes["/prompt_encoder/Pad_1"], nodes["/prompt_encoder/Clip"]
    assert pad.input[0] == "input_labels" and clip.input[0] == pad.output[0], "unexpected prompt encoder"
    constants = {n.output[0]: n for n in graph.node if n.op_type == "Constant"}

    def as_float(name):
        value = numpy_helper.to_array(constants[name].attribute[0].t).astype(np.float32)
        graph.initializer.append(numpy_helper.from_array(value, name + "_float"))
        if sum(name in n.input for n in graph.node) == 1:
            graph.node.remove(constants[name])
        return name + "_float"

    index = clip.output[0]
    pad.input[0], pad.input[2] = "input_labels_float", as_float(pad.input[2])
    clip.input[1], clip.output[0] = as_float(clip.input[1]), "labels_clipped_float"
    order = list(graph.node)
    graph.node.insert(order.index(pad), helper.make_node(
        "Cast", ["input_labels"], ["input_labels_float"], to=TensorProto.FLOAT, name="labels_to_float"))
    graph.node.insert(list(graph.node).index(clip) + 1, helper.make_node(
        "Cast", ["labels_clipped_float"], [index], to=TensorProto.INT64, name="labels_to_index"))
    stale = {pad.output[0], index}
    kept = [v for v in graph.value_info if v.name not in stale]
    del graph.value_info[:]
    graph.value_info.extend(kept)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def reference(model, feats, points, labels):
    """transformers' own path on the same features: (low-res mask, pointer) per multimask setting."""
    out = {}
    for multimask in (True, False):
        o = model._single_frame_forward(image_embeddings=list(feats), input_points=points, input_labels=labels,
                                        multimask_output=multimask)
        out[multimask] = (o.pred_masks, o.object_pointer)
    return out


def main():
    if sys.argv[1:2] == ["--patch"]:
        labels_as_float(sys.argv[2])
        data = open(sys.argv[2], "rb").read()
        print(sys.argv[2], len(data), "bytes, sha256", hashlib.sha256(data).hexdigest())
        return
    import transformers

    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(".models", "edgetam-decoder")
    os.makedirs(out_dir, exist_ok=True)
    model = transformers.EdgeTamVideoModel.from_pretrained(REPO, revision=REVISION).eval()
    cfg = model.config.mask_decoder_config
    print("stability delta/thresh:", cfg.dynamic_multimask_stability_delta, cfg.dynamic_multimask_stability_thresh)
    decoder = Decoder(model).eval()

    g = torch.Generator().manual_seed(0)
    feats = (torch.randn(1, 32, 256, 256, generator=g), torch.randn(1, 64, 128, 128, generator=g),
             torch.randn(1, 256, 64, 64, generator=g))
    points = torch.tensor([[[[310.0, 10.0], [517.0, 405.0]]]])
    labels = torch.tensor([[[2, 3]]], dtype=torch.int32)
    path = os.path.join(out_dir, "mask_decoder.onnx")
    with torch.no_grad():
        torch.onnx.export(
            decoder, (*feats, points, labels), path, dynamo=False, opset_version=18,
            input_names=["feats0", "feats1", "feats2", "input_points", "input_labels"],
            output_names=["masks", "iou_scores", "object_pointers", "object_score_logits"],
            dynamic_axes={"input_points": {2: "num_points"}, "input_labels": {2: "num_points"}})
    labels_as_float(path)

    # The export against transformers' own decoder, on the CPU, for one, two and three points.
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    cases = [(points, labels),
             (torch.tensor([[[[395.0, 210.0]]]]), torch.tensor([[[1]]], dtype=torch.int32)),
             (torch.tensor([[[[395.0, 210.0], [230.0, 250.0], [600.0, 350.0]]]]),
              torch.tensor([[[1, 0, 1]]], dtype=torch.int32))]
    for p, l in cases:
        with torch.no_grad():
            want = decoder(*feats, p, l)
            ref = reference(model, feats, p, l)
        got = session.run(None, dict(feats0=feats[0].numpy(), feats1=feats[1].numpy(), feats2=feats[2].numpy(),
                                     input_points=p.numpy(), input_labels=l.numpy()))
        diffs = [float(np.abs(a - b.numpy()).max()) for a, b in zip(got, want)]
        masks, iou, pointers, score = want
        best = 1 + int(iou[0, 0, 1:].argmax())
        gate = lambda m: torch.where(score[0, 0] > 0, m, torch.tensor(-1024.0))
        # Multimask: best of three, its pointer. Single mask: token 0's pointer (the stability fallback swaps
        # only the mask), checked here where token 0 is stable or not.
        multi_ok = (torch.allclose(gate(masks[0, 0, best]), ref[True][0][0, 0], atol=1e-4)
                    and torch.allclose(pointers[0, 0, best], ref[True][1][0, 0], atol=1e-4))
        single_ptr_ok = torch.allclose(pointers[0, 0, 0], ref[False][1][0, 0], atol=1e-4)
        print(f"{p.shape[2]} points: ONNX vs torch max diff {max(diffs):.2e}; matches transformers multimask "
              f"{multi_ok}, single-mask pointer {single_ptr_ok}")

    data = open(path, "rb").read()
    print(path, len(data), "bytes, sha256", hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
