#!/usr/bin/env bash
# Start the miner. Builds for this machine's GPU the first time (or if the GPU changed).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
BIN="${ROOT}/build/bin/xnminer"

detect_arch() {
  if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    echo "${CMAKE_CUDA_ARCHITECTURES}"
    return
  fi
  # Same mapping as build.sh (scripts/detect-hardware.sh): Blackwell consumer = 120a.
  if command -v nvidia-smi >/dev/null 2>&1; then
    source "${ROOT}/scripts/detect-hardware.sh"
    if [[ "${XN_DETECT_OK:-0}" == "1" ]]; then
      echo "${XN_BUILD_ARCH}"
      return
    fi
  fi
  # No GPU visible at build time — emit a fat binary for common cards.
  echo "75;86;89;90;120a"
}

need_build=0
if [[ ! -x "${BIN}" ]]; then
  need_build=1
else
  current="$(detect_arch)"
  built="$(tr -d ' \r\n' < "${ROOT}/data/cuda_arch" 2>/dev/null || true)"
  if [[ -z "${built}" || "${built}" != "${current}" ]]; then
    echo "GPU architecture changed (${built:-none} -> ${current}) — rebuilding."
    need_build=1
  fi
fi

if [[ "${need_build}" -eq 1 ]]; then
  echo "Building pure CUDA miner for this GPU..."
  "${ROOT}/build.sh"
fi

if [[ ! -f "${ROOT}/miner.ini" ]]; then
  cp "${ROOT}/miner.ini.example" "${ROOT}/miner.ini"
  echo "Created miner.ini (empty wallet — you will be asked for your 0x address)."
  # Auto-detect: if port 80 (/difficulty) is blocked (common on vast.ai and some
  # hosting networks), match-drain would park the GPU on the fallback oracle.
  # Disable it up front so mining runs at full speed out of the box.
  if curl -s --max-time 8 -o /dev/null http://xenblocks.io/difficulty; then
    echo "Port 80 (/difficulty): OK — match_drain_enabled stays true."
  else
    sed -i 's/^match_drain_enabled = true/match_drain_enabled = false/' "${ROOT}/miner.ini"
    echo "Port 80 (/difficulty): BLOCKED on this network — set match_drain_enabled=false (full-speed mining; queue flushes via lastblock)."
  fi
fi

mkdir -p "${ROOT}/data"
exec "${BIN}" "$@"
