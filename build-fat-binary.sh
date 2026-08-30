#!/usr/bin/env bash
# Buduje JEDEN multi-arch binary (sm_86 + sm_89 + sm_120a) — RTX 3060 / 4070 / 5090.
# Uruchom w katalogu repo xnminer (tam gdzie build.sh) na maszynie z CUDA toolkit (nvcc),
# e.g. on a vast.ai RTX 5090. HiveOS has NO nvcc — you never build there, only copy the binary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# sm_120a wymaga nvcc 12.8+ (CUDA 13 preferowane). Na starszym nvcc: flota (86;89)
# without Blackwell — a 5090 will not work with that binary.
if nvcc --version 2>/dev/null | grep -qE 'release 12\.(8|9)|release 13'; then
  export CMAKE_CUDA_ARCHITECTURES="86;89;120a"
  echo "Building multi-arch fat binary: sm_86 (RTX 3060) + sm_89 (RTX 4070) + sm_120a (RTX 5090 Blackwell)"
else
  export CMAKE_CUDA_ARCHITECTURES="86;89"
  echo "nvcc < 12.8 — building ONLY sm_86 + sm_89 (RTX 30xx/40xx). For a 5090 install CUDA 12.8+/13 (scripts/ensure-cuda13.sh) and rebuild."
fi

./build.sh

BIN="${ROOT}/build/bin/xnminer"
echo
echo "OK: ${BIN}"
echo "The same binary runs on RTX 30xx, 40xx and 50xx cards."
echo "Skopiuj go do pakietu floty:  cp ${BIN} /sciezka/do/xnminer-fleet/bin/xnminer"
if command -v cuobjdump >/dev/null 2>&1; then
  echo
  echo "Wkompilowane architektury:"
  cuobjdump -lelf "${BIN}" || true
fi
