# Parity with PyTorch

Both ready-made ONNX video trackers, driven from Rust (this crate) on ONNX Runtime's CUDA provider,
against transformers' PyTorch models (`EdgeTamVideoModel`, `Sam2VideoModel`) on EdgeTAM's bedroom video.

## How to reproduce

```bash
# The models at pinned revisions, and CUDA 13 + cuDNN 9 from NVIDIA's wheels (no CUDA toolkit; track-parity loads
# them from .models/cuda-runtime, or --cuda <dir>)
scripts/fetch_models.sh
python -I scripts/parity/reference.py <EdgeTAM>/notebooks/videos/bedroom .cache/reference --model edgetam   # CUDA torch
python -I scripts/export/edgetam_decoder.py   # optional: our decoder, into .models/edgetam-decoder (used when present)
cargo run --release --bin track-parity -- .cache/reference/edgetam --models .models/edgetam-video --tracker edgetam
```

The sequences are edgetam-coreml's: five runs of 40 frames, three objects seeded on frame 0, one on frame 120, and one
with 8 black frames in the middle where the object leaves and comes back. Each side carries its own memory; the bound
is a mask IoU of at least 0.95 on every frame, at the video's size.

## Accuracy (2026-10-19, RTX 3070, ORT 1.2x CUDA EP, fp32 with TF32)

| Sequence | EdgeTAM IoU min / mean | SAM 2.1 tiny IoU min / mean |
|---|---|---|
| bedroom-girl (2 points) | 0.968 (frame 0) / 0.988 | 0.981 / 0.995 |
| bedroom-boy | **0.908** (frame 12) / 0.974 | **0.948** (frame 31) / 0.992 |
| bedroom-pillow | 0.997 / 0.998 | 0.991 / 0.998 |
| bedroom-boy-late (from frame 120) | 0.955 / 0.991 | 0.987 / 0.996 |
| bedroom-girl-gap (8 black frames) | 0.968 / 0.991 | 0.981 / 0.995 |
| Within bounds | 4 of 5 | 4 of 5 |

Both detect the object leaving on exactly the black frames 16–23, as PyTorch does.

What differs from PyTorch:
- **EdgeTAM's mask decoder is exported with multimask on.** PyTorch takes the single-mask path for two or more
  points; the export always picks the best of three. That is the 0.968 on the girl's seed frame (a click plus a
  negative point). One click, and every tracked frame, take the same path on both sides. Fixed by a decoder of our
  own, below.
- **SAM 2.1's bank is padded.** Its memory attention has fixed shapes, so early frames repeat the newest memory to
  fill 7 slots and 16 pointers; PyTorch attends only the frames it has.
- Our preprocessing (Rust, triangle-filter resize) is not byte-identical to transformers'.

## Our EdgeTAM decoder

`scripts/export/edgetam_decoder.py` exports the prompt encoder and mask decoder from transformers'
`EdgeTamVideoModel` (`yonigozlan/EdgeTAM-hf` at `c266ce53b3fc`, Apache-2.0) with the published file's inputs, but
returns all four masks (the single-mask token and the three multimask candidates), their predicted IoUs, one gated
object pointer per token and the object score. The `select_mask` glue graph then picks as transformers does: the best
of three for one point or none (every tracked frame), otherwise the single mask with
`_dynamic_multimask_via_stability` (delta 0.05, threshold 0.98) and token 0's pointer. `EdgeTam` uses it when
`.models/edgetam-decoder/mask_decoder.onnx` exists (next to `edgetam-video`), and the published decoder otherwise.

- File: `mask_decoder.onnx`, 21,906,087 bytes, SHA-256
  `90265335fd5f8bb7c65f2058ff961ea13025eafa2f874a766760869b7d461a89` (torch 2.14.1, transformers 5.20.1, opset 18).
  Not hosted yet; produce it with the script (CUDA torch venv) and it is picked up.
- The export pads and clamps the point labels in float (`labels_as_float`): ORT has no CUDA kernels for
  int32 Pad and Clip, and with those two on the CPU the decoder could not be a CUDA graph. The outputs are exactly
  the first export's (21,906,013 bytes, `e7d6f452…`); `--patch <file>` converts that one in place. With an
  unpatched file the tracking decoder falls back to a plain session.
- Checked on the CPU against transformers' own `_single_frame_forward`, with one, two and three points, with
  stable and unstable single masks and with an absent object: mask logits within 8e-5 and pointers within 4e-6.

The reference harness had its own difference: `reference.py` gave PyTorch the source JPEG's pixels but saved the
frames re-encoded at quality 95 for the Rust side. It now gives PyTorch the frames as saved (the "fixed" references
below). Min IoU over 40 frames (bold: within 0.95). "strict" is PyTorch with TF32 off for cuDNN convolutions too
(PyTorch's default leaves it on):

| Sequence | Published, old refs | Ours, old refs | Published, fixed refs | Ours, fixed refs | Ours vs strict | PyTorch default vs strict |
|---|---|---|---|---|---|---|
| bedroom-girl (2 points) | **0.967** | **0.993** | **0.967** | **0.980** | **0.980** | 0.999 |
| bedroom-boy (2 points) | 0.908 | 0.851 | 0.794 | 0.916 | **0.992** | 0.916 |
| bedroom-pillow | **0.996** | **0.996** | **0.998** | **0.998** | **0.998** | 0.9995 |
| bedroom-boy-late | **0.965** | **0.965** | 0.911 | 0.911 | **0.963** | 0.916 |
| bedroom-girl-gap | **0.967** | **0.980** | **0.967** | **0.978** | **0.978** | 0.999 |
| bedroom-girl-box | 0.795 | 0.934 | 0.778 | **0.969** | 0.932 | 0.931 |
| bedroom-boy-box | 0.897 | 0.932 | 0.933 | 0.900 | **0.993** | 0.899 |

Seed frames now agree: girl box 0.907 → 0.993, boy box 0.953 → 0.992, the girl's click plus negative point
0.967 → 0.993. Speed is unchanged (~46 ms per tracked frame).

**Why the bound is not met on every sequence.** PyTorch fails it against itself. With only TF32 convolutions toggled
it scores 0.916 (boy, frame 4), 0.916 (boy-late, frame 15), 0.899 (boy box, frame 9) and 0.931 (girl box, frame 21).
These sequences have frames where the decoder's top two candidates are within 0.0006–0.008 in predicted IoU (boy box
frames 5–6: 0.9338 vs 0.9307 and 0.9382 vs 0.9376), or where the mask is uncertain anyway (frames 9 and 21 predict
under 0.91 for every candidate). Any rounding difference can flip which mask is kept. The flipped mask then goes into
the memory, so the run follows one of two diverging trajectories. Ours is deterministic run to run. On every sequence
it is within 0.95 of one of PyTorch's two runs: 0.969 (girl box, default), 0.993 (boy box, strict), 0.992 (boy,
strict). Matmul TF32 alone does not move PyTorch (0.9997 against its default); ORT's TF32 switch moves ours from one
trajectory to the other (girl box 0.969 → 0.935, boy box 0.900 → 0.944), so it stays on for speed. Against strict
PyTorch, our decoder passes all 5 click sequences, where the published one passes 4 (boy 0.793).

## EdgeTAM on the GPU

`EdgeTam` (`src/edgetam.rs`) keeps every per-frame tensor on the GPU. Only the decoded frame
goes up (H×W×3 bytes) and only the mask at the video's size and the object score come back.

- **IoBinding throughout.** The model's four graphs pass tensors through fixed device buffers. A binding keeps its
  own reference to an input, so each buffer is bound as the reader's input first and then handed to the writer's
  output.
- **Glue graphs for the steps in between** (`data/glue/*.onnx`, from `scripts/glue/make_glue.py`, checked against
  numpy): resize and normalise the frame (antialiased bilinear, like transformers' fast processor), layouts, the
  seed's no-memory embedding, the bank's positions, the mask at the video's size.
- **The bank is a ring on the device.** Spatial slot 0 is the seed, slots 1–6 the six latest frames; pointer slot 0
  the seed, then the latest frames. The memory attention reads a view of the filled slots. Block order does not
  matter to it (rotary positions repeat per memory, pointers carry none), so only each slot's temporal row changes
  per frame, and `bank_positions` adds it from a list of seven ages. A new memory goes into its slot with a
  device-to-device copy.
- **CUDA graphs** on the vision encoder and the tracking memory encoder: one launch replays all their kernels. The
  encoder's position outputs are constants that ORT folds, and copying a constant out cannot be recorded, so they
  are taken once from a plain session at load. Exhaustive cuDNN search and on-demand workspace are off for graph
  sessions; both allocate or benchmark during the first run, which a graph cannot record. The published mask decoder
  has control-flow nodes and stays a plain session on the same fixed buffers; ours is a graph too, with its point
  count fixed at one (tracked frames have only the padding point) so that ORT folds its shape arithmetic. The memory
  attention's input grows with the bank, so it is a plain session until the bank is full (16 frames after the seed)
  and a graph from then on, reading a stage the object's bank is copied into (0.9 MB, device to device).
- **Several objects, one encoder pass**. `EdgeTam::encode` runs the preprocessing, the encoder and the
  sequence layout once per frame; `add` seeds an object on it and `step` moves one to it. An object owns only its
  bank: the graphs and fixed buffers are shared, since objects step one after another. Banks come from a plain
  session's allocator: the CUDA graph sessions' allocators may hand out memory their recorded graphs replay into
  (banks allocated from the encoder's after its first run were overwritten every frame).

### Speed (2026-10-19, RTX 3070 8 GB, every CPU core busy with other processes)

| Pipeline | ms per tracked frame |
|---|---|
| Host round trips between graphs (first version) | 130–1000, varying run to run |
| GPU-resident, no CUDA graphs | ~130 |
| GPU-resident, encoder as a CUDA graph | **46–51**, steady across all five sequences |
| Also our decoder, and the memory attention once the bank is full, as CUDA graphs | **41–42** |

The vision encoder went from 86 to 14 ms once it was a CUDA graph: GPU utilisation had been 5 %, the GPU waiting
on a starved CPU to launch each kernel. What remains per frame: memory attention ~13 ms (its input grows until the
bank is full, so it is not a graph), decoder ~8 ms, memory encoder ~7 ms, glue under 2 ms in all. On
full-bank frames: encoder 15 ms once per frame; per object memory attention 11.6 ms (GPU-bound: a graph saves under
2 ms), decoder 2.1 ms, memory encoder 4.7 ms, mask choice, layout and mask out 2 ms. Masks are byte-identical to
before on every sequence of both reference sets.

Accuracy of the GPU pipeline against PyTorch EdgeTAM: IoU min / mean 0.967 / 0.988 (girl), 0.908 / 0.976 (boy),
0.996 / 0.998 (pillow), 0.965 / 0.995 (boy, late), 0.967 / 0.990 (gap; absent on exactly frames 16–23). 4 of 5 within
bounds, as the host version.

## Choice

EdgeTAM: exact bank (no padding), mobile RepViT encoder, memory attention over only the frames that exist, and
about 50 ms a frame on an RTX 3070. SAM 2.1 tiny stays as the fallback behind the same trait (host-side driver, its
accuracy here slightly better on the boy sequence).

Open: hosting our decoder, so it can be pinned like the ready-made export. The remaining misses are
frames where PyTorch disagrees with itself at the 0.95 bound (above), so a per-frame bound of 0.95 cannot hold on
them. The trajectory spread is the measure there.
