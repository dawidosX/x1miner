#!/usr/bin/env bash
# Install a Blackwell-capable CUDA *toolkit* (nvcc + cudart) without touching
# the NVIDIA driver — and match it to the HOST DRIVER, which the user cannot
# change (vast.ai / hosting):
#   driver CUDA >= 13.0  ->  CUDA 13.x toolkit
#   driver CUDA 12.8/12.9 -> CUDA 12.8/12.9 toolkit (sm_120a supported since 12.8)
#   driver CUDA < 12.8   ->  Blackwell impossible on this host (clear message)
# A binary built with a newer runtime than the driver DIES at cudaSetDevice
# ("driver version is insufficient"). Never apt-install cuda / cuda-drivers.
set -euo pipefail

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

driver_cuda_version() {
  # nvidia-smi header prints e.g. "CUDA Version: 12.8"
  command -v nvidia-smi >/dev/null 2>&1 || { echo ""; return; }
  nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1
}

nvcc_major() {
  local bin="$1"
  [[ -x "${bin}" ]] || return 1
  "${bin}" --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\).*/\1/p' | head -n1
}

find_cuda13_home() {
  local d
  for d in /usr/local/cuda-13.6 /usr/local/cuda-13.5 /usr/local/cuda-13.4 /usr/local/cuda-13.3 \
           /usr/local/cuda-13.2 /usr/local/cuda-13.1 /usr/local/cuda-13.0 \
           /usr/local/cuda; do
    local maj
    maj="$(nvcc_major "${d}/bin/nvcc" || true)"
    if [[ "${maj}" == "13" ]]; then
      echo "${d}"
      return 0
    fi
  done
  local which_nvcc
  which_nvcc="$(command -v nvcc 2>/dev/null || true)"
  if [[ -n "${which_nvcc}" ]]; then
    local maj
    maj="$(nvcc_major "${which_nvcc}" || true)"
    if [[ "${maj}" == "13" ]]; then
      echo "$(cd "$(dirname "${which_nvcc}")/.." && pwd)"
      return 0
    fi
  fi
  return 1
}

ubuntu_cuda_repo() {
  local ver="22.04"
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    ver="${VERSION_ID:-22.04}"
  fi
  case "${ver}" in
    24.04) echo ubuntu2404 ;;
    26.04) echo ubuntu2604 ;;
    20.04) echo ubuntu2004 ;;
    *) echo ubuntu2204 ;;
  esac
}

add_nvidia_cuda_repo() {
  command -v apt-get >/dev/null 2>&1 || return 1
  if [[ -f /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list ]] ||
     [[ -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]] ||
     [[ -f /etc/apt/sources.list.d/cuda.list ]] ||
     ls /etc/apt/sources.list.d/cuda*.list >/dev/null 2>&1; then
    return 0
  fi
  local repo
  repo="$(ubuntu_cuda_repo)"
  local deb="/tmp/cuda-keyring_1.1-1_all.deb"
  local url="https://developer.download.nvidia.com/compute/cuda/repos/${repo}/x86_64/cuda-keyring_1.1-1_all.deb"
  echo "Adding NVIDIA CUDA apt repo (${repo})..."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "${deb}" "${url}" || return 1
  else
    wget -q -O "${deb}" "${url}" || return 1
  fi
  as_root dpkg -i "${deb}" >/dev/null
  as_root apt-get update -y
}

install_cuda_packages() {  # args: package version suffixes, e.g. 13-3 12-9 12-8
  command -v apt-get >/dev/null 2>&1 || {
    echo "ensure-cuda13: apt-get not found; install CUDA 13 nvcc by hand." >&2
    return 1
  }
  export DEBIAN_FRONTEND=noninteractive
  # Never pull a driver stack. Hold common driver packages just in case.
  # Do not install nvidia-driver / cuda-drivers — that desyncs libcuda from
  # the host driver and nvidia-smi dies until reboot.
  as_root apt-mark hold nvidia-driver nvidia-driver-580 nvidia-driver-570 \
    nvidia-driver-610 nvidia-open cuda-drivers cuda-drivers-580 cuda-drivers-570 2>/dev/null || true

  add_nvidia_cuda_repo || true
  as_root apt-get update -y

  local ver pkgs
  for ver in "$@"; do
    pkgs=(
      "cuda-nvcc-${ver}"
      "cuda-cudart-dev-${ver}"
      "cuda-cccl-${ver}"
    )
    echo "Trying CUDA toolkit packages: ${pkgs[*]}"
    if as_root apt-get install -y --no-install-recommends "${pkgs[@]}"; then
      echo "Installed CUDA ${ver} toolkit (compiler only)."
      return 0
    fi
    echo "Package set cuda-nvcc-${ver} not available; trying cuda-toolkit-${ver}"
    if as_root apt-get install -y --no-install-recommends "cuda-toolkit-${ver}"; then
      echo "Installed cuda-toolkit-${ver} (compiler + libs, no driver meta)."
      return 0
    fi
  done
  return 1
}

export_cuda13() {
  local home="$1"
  export CUDA_HOME="${home}"
  export CMAKE_CUDA_COMPILER="${home}/bin/nvcc"
  export CUDAToolkit_ROOT="${home}"
  export PATH="${home}/bin:${PATH}"
  export LD_LIBRARY_PATH="${home}/lib64:${LD_LIBRARY_PATH:-}"
  echo "Using ${home}/bin/nvcc ($("${home}/bin/nvcc" --version | tr '\n' ' '))"
}

ensure_cuda13() {
  local drv drv_major drv_minor
  drv="$(driver_cuda_version)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU driver (host-injected, not upgraded): $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1) (CUDA ${drv:-unknown})"
  fi

  # Pick toolkit versions COMPATIBLE with the host driver. A runtime newer than
  # the driver fails at cudaSetDevice — the user cannot upgrade a hosted driver.
  local homes=() pkgs=()
  drv_major="${drv%%.*}"; drv_minor="${drv#*.}"
  if [[ -z "${drv}" || "${drv_major}" -ge 13 ]]; then
    homes=(13.6 13.5 13.4 13.3 13.2 13.1 13.0)
    pkgs=(13-6 13-5 13-4 13-3 13-2 13-1 13-0)
  elif [[ "${drv_major}" -eq 12 && "${drv_minor}" -ge 8 ]]; then
    echo "Host driver supports CUDA ${drv} — selecting a 12.x toolkit (sm_120a needs >= 12.8)."
    if [[ "${drv_minor}" -ge 9 ]]; then homes=(12.9 12.8); pkgs=(12-9 12-8); else homes=(12.8); pkgs=(12-8); fi
  else
    echo "ensure-cuda13: host driver supports only CUDA ${drv:-<12.8} — Blackwell (sm_120a) needs 12.8+." >&2
    echo "This host cannot build/run a Blackwell binary; on RTX 20xx-40xx the normal build works fine." >&2
    return 1
  fi

  local home
  if home="$(find_cuda_home "${homes[@]}")"; then
    export_cuda13 "${home}"
    return 0
  fi
  echo "No compatible nvcc found. Installing CUDA toolkit matching the driver (will not change the GPU driver)..."
  if ! install_cuda_packages "${pkgs[@]}"; then
    echo "ensure-cuda13: FAILED to install a driver-compatible CUDA toolkit." >&2
    return 1
  fi
  if home="$(find_cuda_home "${homes[@]}")"; then
    export_cuda13 "${home}"
    return 0
  fi
  echo "ensure-cuda13: packages installed but a matching nvcc is not on disk." >&2
  return 1
}

ensure_cuda13
