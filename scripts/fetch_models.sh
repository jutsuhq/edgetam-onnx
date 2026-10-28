#!/usr/bin/env bash
# Downloads the trackers' ONNX exports at pinned revisions into .models/, and CUDA 13 + cuDNN 9 from NVIDIA's pip
# wheels into .models/cuda-runtime (one flat folder, as cuda::use_runtime wants it). Needs curl and pip.
#
#     scripts/fetch_models.sh            # EdgeTAM, SAM 2.1 tiny and the CUDA runtime
#     scripts/fetch_models.sh --no-cuda  # the models only
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models="$root/.models"

fetch() { # <repo> <revision> <folder> <file in repo>...
    local repo=$1 rev=$2 dir="$models/$3"
    shift 3
    mkdir -p "$dir"
    for f in "$@"; do
        [[ -f "$dir/$(basename "$f")" ]] || curl -fL --progress-bar -o "$dir/$(basename "$f")" \
            "https://huggingface.co/$repo/resolve/$rev/$f"
    done
}

fetch jax-image-tools/edgetam-video-onnx 8ca3d3e4169938e65b552cdf14542fde25e8badb edgetam-video \
    vision_encoder.onnx mask_decoder.onnx memory_attention.onnx memory_encoder.onnx
fetch square-zero-labs/sam2.1-tiny-video-onnx 3b2984dd865f6e9d2cc6aed0be6a5a5c2eb352ce sam21-tiny-video \
    onnx/vision_encoder.onnx onnx/mask_decoder.onnx onnx/memory_attention.onnx onnx/memory_encoder.onnx \
    onnx/pointer_tpos.onnx constants.json

[[ "${1:-}" == "--no-cuda" ]] && exit 0
# The versions ONNX Runtime 1.2x's CUDA provider needs (CUDA >= 13.2, cuDNN >= 9.23).
wheels="$root/.cache/cuda-wheels"
pip install -q --no-deps --target "$wheels" nvidia-cuda-runtime==13.4.92 nvidia-cuda-nvrtc==13.4.92 \
    nvidia-nvjitlink==13.4.92 nvidia-cublas==13.8.0.4 nvidia-cufft==12.4.0.43 nvidia-curand==10.4.4.72 \
    nvidia-cudnn-cu13==9.27.0.42
mkdir -p "$models/cuda-runtime"
find "$wheels" -type f \( -name '*.dll' -o -name '*.so' -o -name '*.so.*' \) ! -name '*nvblas*' ! -name '*cufftw*' \
    -exec cp -f {} "$models/cuda-runtime/" \;
echo "models and CUDA runtime in $models"
