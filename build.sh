#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to build cyflow. Install it first: https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [[ "${CYFLOW_ENABLE_CUDA:-0}" =~ ^(1|true|yes|on)$ ]]; then
    echo "Building with CUDA support enabled. Make sure nvcc and the CUDA toolkit are installed." >&2
else
    echo "Building CPU-compatible extension by default. Set CYFLOW_ENABLE_CUDA=1 to attempt a CUDA build on a GPU machine." >&2
fi

uv run --with "setuptools>=83.0.0" --with "Cython>=3.2.8" --with "numpy>=2.5.1" \
    python setup.py build_ext --inplace