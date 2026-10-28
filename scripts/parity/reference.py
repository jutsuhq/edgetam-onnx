"""Reference masks for the tracker parity check: transformers' PyTorch video models on EdgeTAM's bedroom video.

    python -I scripts/parity/reference.py <bedroom JPEG folder> <out folder> [--model edgetam|sam21] [--frames 40]

Writes <out>/<model>/<sequence>/frames/00000.jpg… (the frames both sides see, black ones included),
masks/00000.png… (the object in white, at the video's size) and sequence.json (points, object scores). The Rust
side (`cargo run --release --bin track-parity`) runs the ONNX trackers on the same frames and compares.
The sequences are edgetam-coreml's parity set.
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

MODELS = {
    "edgetam": ("EdgeTamVideoModel", "yonigozlan/EdgeTAM-hf", "c266ce53b3fc"),
    "sam21": ("Sam2VideoModel", "facebook/sam2.1-hiera-tiny", "de431c404385"),
}

# (name, first frame, black frames (from, count) or None, points (x, y, label) in the video's pixels)
SEQUENCES = [
    ("bedroom-girl", 0, None, [(395, 210, 1), (230, 250, 0)]),
    ("bedroom-boy", 0, None, [(230, 250, 1), (395, 210, 0)]),
    ("bedroom-pillow", 0, None, [(600, 350, 1)]),
    ("bedroom-boy-late", 120, None, [(230, 250, 1)]),
    ("bedroom-girl-gap", 0, (16, 8), [(395, 210, 1), (230, 250, 0)]),
    # Seeded by a box, as prep seeds from a person detector; corners carry labels 2 and 3. Boxes from RT-DETRv2.
    ("bedroom-girl-box", 0, None, [(310, 10, 2), (517, 405, 3)]),
    ("bedroom-boy-box", 0, None, [(144, 134, 2), (294, 410, 3)]),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bedroom")
    parser.add_argument("out")
    parser.add_argument("--model", default="edgetam", choices=MODELS)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--only", default="", help="comma-separated sequence names")
    args = parser.parse_args()

    import transformers

    # The GPU does the work; the CPU only feeds it.
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA torch: uv pip install --index-url https://download.pytorch.org/whl/cu130 torch")
    device = "cuda"

    class_name, repo, revision = MODELS[args.model]
    model = getattr(transformers, class_name).from_pretrained(repo, revision=revision).eval().to(device)
    processor = transformers.Sam2VideoProcessor.from_pretrained(repo, revision=revision)
    names = sorted(f for f in os.listdir(args.bedroom) if f.endswith(".jpg"))

    only = set(args.only.split(",")) if args.only else None
    for name, start, gap, points in SEQUENCES:
        if only and name not in only:
            continue
        folder = os.path.join(args.out, args.model, name)
        os.makedirs(os.path.join(folder, "frames"), exist_ok=True)
        os.makedirs(os.path.join(folder, "masks"), exist_ok=True)
        frames = []
        for i, file in enumerate(names[start:start + args.frames]):
            image = Image.open(os.path.join(args.bedroom, file)).convert("RGB")
            if gap and gap[0] <= i < gap[0] + gap[1]:
                image = Image.new("RGB", image.size)
            path = os.path.join(folder, "frames", f"{i:05d}.jpg")
            image.save(path, quality=95)
            # The frame as saved, so both sides start from the same JPEG.
            frames.append(Image.open(path).convert("RGB"))
        width, height = frames[0].size

        with torch.no_grad():
            session = processor.init_video_session(video=frames, inference_device=device, dtype=torch.float32)
            if [p[2] for p in points] == [2, 3]:
                (x0, y0, _), (x1, y1, _) = points
                processor.add_inputs_to_inference_session(
                    inference_session=session, frame_idx=0, obj_ids=1,
                    input_boxes=[[[float(x0), float(y0), float(x1), float(y1)]]])
            else:
                processor.add_inputs_to_inference_session(
                    inference_session=session, frame_idx=0, obj_ids=1,
                    input_points=[[[[float(x), float(y)] for x, y, _ in points]]],
                    input_labels=[[[int(label) for _, _, label in points]]])
            scores = {}

            def save(frame_idx, output):
                logits = processor.post_process_masks(
                    [output.pred_masks], original_sizes=[[height, width]], binarize=False)[0]
                mask = (logits[0, 0].cpu().numpy() > 0).astype(np.uint8) * 255
                Image.fromarray(mask).save(os.path.join(folder, "masks", f"{frame_idx:05d}.png"))
                score = getattr(output, "object_score_logits", None)
                scores[frame_idx] = None if score is None else float(score.ravel()[0])

            save(0, model(inference_session=session, frame_idx=0))
            for output in model.propagate_in_video_iterator(session):
                if output.frame_idx > 0:
                    save(output.frame_idx, output)

        with open(os.path.join(folder, "sequence.json"), "w") as f:
            json.dump(dict(model=args.model, repo=repo, revision=revision, sequence=name, start=start, gap=gap,
                           width=width, height=height, frames=len(frames),
                           points=[[x, y, label] for x, y, label in points],
                           object_scores=[scores.get(i) for i in range(len(frames))]), f, indent=1)
        print(f"{args.model} {name}: {len(frames)} frames", flush=True)


if __name__ == "__main__":
    main()
