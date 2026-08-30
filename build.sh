#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${ROOT}/build"
mkdir -p "${ROOT}/data"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found. Run ./install-deps.sh first." >&2
  exit 1
fi

# GPU / CPU / VRAM → arch (Blackwell consumer = 120a), lanes, compiler choice.
# shellcheck disable=SC1091
source "${ROOT}/scripts/detect-hardware.sh"
xn_hardware_report

if [[ "${XN_UNSUPPORTED}" == "1" ]]; then
  echo "ERROR: ${XN_UNSUPPORTED_REASON}" >&2
  echo "This miner targets Turing (RTX 20 / GTX 16) and newer NVIDIA GPUs." >&2
  exit 1
fi

# Blackwell: CUDA 13 nvcc matches the fast desktop SASS. Do not install it on
# Ada/Ampere — a CUDA 13 binary needs driver 580, which those boxes often lack.
if [[ "${XN_NEED_CUDA13}" == "1" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/ensure-cuda13.sh" || true
  if command -v nvcc >/dev/null 2>&1 && ! nvcc --version 2>/dev/null | grep -q 'release 13'; then
    echo "WARNING: Blackwell GPU but nvcc is not CUDA 13. H/s will be lower. Continuing." >&2
  fi
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Install the NVIDIA CUDA Toolkit and put it on PATH." >&2
  echo "  export PATH=/usr/local/cuda/bin:\$PATH" >&2
  exit 1
fi

NVCC_VER="$(nvcc --version 2>/dev/null | tr '\n' ' ' || true)"
echo "nvcc: ${NVCC_VER}"

nvcc_has_blackwell() {
  nvcc --version 2>/dev/null | grep -qE 'release 12\.(8|9)|release 13'
}

ARCH="${XN_BUILD_ARCH}"
if [[ "${ARCH}" == "75;86;89;90;120a" || "${ARCH}" == "75;86;89;90;120" ]] && ! nvcc_has_blackwell; then
  ARCH="75;86;89;90"
  echo "nvcc is older than 12.8 — fat cubin without sm_120a"
fi
if [[ "${ARCH}" == *"120"* || "${ARCH}" == *"100"* ]] && ! nvcc_has_blackwell; then
  echo "ERROR: this GPU needs nvcc 12.8+ (CUDA 13 preferred, Blackwell 120a). Have: ${NVCC_VER}" >&2
  echo "Install a CUDA devel image with 12.8/13, or run scripts/ensure-cuda13.sh" >&2
  exit 1
fi

echo "Building xnminer (Linux CUDA) CMAKE_CUDA_ARCHITECTURES=${ARCH}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv || true
fi

GEN=()
if command -v ninja >/dev/null 2>&1; then
  GEN=(-G Ninja)
fi

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"
)
if [[ -n "${CMAKE_CUDA_COMPILER:-}" ]]; then
  CMAKE_ARGS+=(-DCMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER}")
fi
if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then
  CMAKE_ARGS+=(-DCUDAToolkit_ROOT="${CUDAToolkit_ROOT}")
fi

rm -f "${BUILD}/CMakeCache.txt"
cmake -S "${ROOT}" -B "${BUILD}" "${GEN[@]}" "${CMAKE_ARGS[@]}"
cmake --build "${BUILD}" --parallel

echo "${ARCH}" > "${ROOT}/data/cuda_arch"

echo
if command -v cuobjdump >/dev/null 2>&1 && [[ -x "${BUILD}/bin/xnminer" ]]; then
  echo "Embedded GPU code:"
  cuobjdump -lelf "${BUILD}/bin/xnminer" || true
fi
echo "OK: ${BUILD}/bin/xnminer"
