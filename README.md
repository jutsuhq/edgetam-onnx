<div align="center">

# edgetam-onnx

**Click once, track the object through video: EdgeTAM and SAM 2.1 tiny in Rust on ONNX Runtime's CUDA provider.**

~42 ms a frame on an RTX 3070, masks within 0.95 IoU of PyTorch, no Python and no CUDA toolkit at run time.

[![License: MIT](https://img.shields.io/badge/License-MIT-F2FF59.svg?labelColor=211927)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2024-000000?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-CUDA-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-211927)](#requirements)

</div>

The Windows and Linux sibling of [edgetam-coreml](https://github.com/jutsuhq/edgetam-coreml), and the tracker inside
[Kawarimi](https://github.com/jutsuhq/kawarimi-local-character-swap).

- **EdgeTAM** ([`jax-image-tools/edgetam-video-onnx`](https://huggingface.co/jax-image-tools/edgetam-video-onnx)):
  every per-frame tensor stays on the GPU. The model's four graphs are chained through IoBinding, small glue graphs
  of our own do the steps between them, and the encoder runs as a CUDA graph. Several objects share one encoder
  pass per frame.
- **SAM 2.1 tiny** ([`square-zero-labs/sam2.1-tiny-video-onnx`](https://huggingface.co/square-zero-labs/sam2.1-tiny-video-onnx)):
  the fallback behind the same trait, with the memory bank on the host.
- Seeds from points (positive, negative) or a box, as transformers' `*VideoModel` takes them.
- An optional export of EdgeTAM's decoder with all four masks, so the mask is chosen as transformers chooses it.

## Use

```toml
[dependencies]
edgetam-onnx = { git = "https://github.com/jutsuhq/edgetam-onnx" }
```

```rust
use std::path::Path;
use edgetam_onnx::{Device, VideoTracker, cuda, edgetam::EdgeTam};

cuda::use_runtime(Path::new(".models/cuda-runtime"))?; // CUDA 13 + cuDNN 9, once per process
let mut tracker = EdgeTam::load(Path::new(".models/edgetam-video"), Device::Cuda, frames.len())?;

// Frames are RGB, 8-bit, interleaved, at the video's size. Label 1 is a positive click.
let first = tracker.seed(&frames[0], width, height, &[[420.0, 310.0]], &[1])?;
for rgb in &frames[1..] {
    let step = tracker.track(rgb, width, height)?;
    // step.mask: 1 where the object is; step.object_score > 0 while it is in view
}
```

For several objects at once: `EdgeTam::encode` per frame, then `add` to seed an object and `step` to move each one.

## Models and CUDA

```bash
scripts/fetch_models.sh
```

Downloads both exports at pinned revisions into `.models/`, and CUDA 13 + cuDNN 9 from NVIDIA's pip wheels into
`.models/cuda-runtime`. `cuda::use_runtime` loads them by full path, so there is no PATH, LD_LIBRARY_PATH or CUDA
toolkit to set up; only the NVIDIA driver.

## Accuracy and speed

Against transformers' PyTorch models on EdgeTAM's bedroom video, five 40-frame sequences (RTX 3070, fp32 with TF32):

| Sequence | EdgeTAM IoU min / mean | SAM 2.1 tiny IoU min / mean |
|---|---|---|
| girl (2 points) | 0.968 / 0.988 | 0.981 / 0.995 |
| boy | 0.908 / 0.974 | 0.948 / 0.992 |
| pillow | 0.997 / 0.998 | 0.991 / 0.998 |
| boy from frame 120 | 0.955 / 0.991 | 0.987 / 0.996 |
| girl, 8 black frames | 0.968 / 0.991 | 0.981 / 0.995 |

| EdgeTAM pipeline | ms per frame |
|---|---|
| Host round trips between graphs | 130–1000 |
| GPU-resident | ~130 |
| GPU-resident, encoder as a CUDA graph | 46–51 |
| Also decoder and full-bank memory attention as CUDA graphs | **41–42** |

How it was measured and how to reproduce it: [docs/PARITY.md](docs/PARITY.md).

```bash
python -I scripts/parity/reference.py <EdgeTAM>/notebooks/videos/bedroom .cache/reference --model edgetam
cargo run --release --bin track-parity -- .cache/reference/edgetam --models .models/edgetam-video --tracker edgetam
```

## Requirements

- Windows or Linux, x86_64, an NVIDIA GPU with a CUDA 13 driver (Linux: 580+)
- `--cpu` / `Device::Cpu` runs anywhere ONNX Runtime does, slowly

## Scripts

| Script | What it does |
|---|---|
| `scripts/fetch_models.sh` | Models and CUDA runtime |
| `scripts/export/edgetam_decoder.py` | EdgeTAM's decoder with all four masks, from transformers |
| `scripts/glue/make_glue.py` | Rebuilds `data/glue/*.onnx` |
| `scripts/parity/edgetam_constants.py` | Rebuilds `data/edgetam_constants.json` from the reference model |
| `scripts/parity/reference.py` | PyTorch reference masks for the parity check |

## License

[MIT](LICENSE). The models keep their own licenses: EdgeTAM and SAM 2.1 are Apache-2.0.
