#!/usr/bin/env bash
set -euo pipefail

echo "Installing host build deps for xnminer (Linux CUDA)..."

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libcurl4-openssl-dev
elif command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y gcc-c++ cmake ninja-build pkgconf-pkg-config libcurl-devel
elif command -v pacman >/dev/null 2>&1; then
  sudo pacman -S --needed --noconfirm base-devel cmake ninja pkgconf curl
else
  echo "Unknown package manager. Install: g++, cmake, ninja, pkg-config, libcurl." >&2
  exit 1
fi

echo
echo "Host deps OK."
echo "You still need:"
echo "  1. An NVIDIA driver (nvidia-smi lists your GPU)"
echo "  2. CUDA Toolkit with nvcc (https://developer.nvidia.com/cuda-downloads)"
echo
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
else
  echo "nvidia-smi not found yet — install the NVIDIA driver before mining."
fi
nproc 2>/dev/null | awk '{print "CPU logical processors: "$1}' || true
