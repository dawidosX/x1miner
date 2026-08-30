#!/usr/bin/env bash
# Probe this box: GPU name / SM / VRAM, CPU cores.
# Sets XN_* variables when sourced. Prints a report when executed.
#
#   source scripts/detect-hardware.sh
#   bash scripts/detect-hardware.sh
#
# Override GPU with DEVICE=N (default 0). Override arch with CMAKE_CUDA_ARCHITECTURES.

xn_trim() { printf '%s' "$1" | tr -d '[:space:]'; }

xn_family_from_arch() {
  local arch="$1"
  case "${arch}" in
    120|120a|121|100|101|103) echo "Blackwell" ;;
    90) echo "Hopper" ;;
    89) echo "Ada" ;;
    87) echo "Orin" ;;
    86|80) echo "Ampere" ;;
    75) echo "Turing" ;;
    70|72) echo "Volta" ;;
    61|60|62) echo "Pascal" ;;
    52|53) echo "Maxwell" ;;
    *) echo "CUDA" ;;
  esac
}

xn_suggested_lanes() {
  local mib="${1:-0}"
  if [[ "${mib}" -ge 114688 ]]; then echo 32
  elif [[ "${mib}" -ge 57344 ]]; then echo 16
  elif [[ "${mib}" -ge 28672 ]]; then echo 8
  elif [[ "${mib}" -ge 14336 ]]; then echo 4
  elif [[ "${mib}" -ge 7168 ]]; then echo 2
  else echo 1
  fi
}

xn_suggested_keygen() {
  local cores="${1:-1}"
  local n=$((cores / 2))
  if [[ "${n}" -lt 2 ]]; then n=2; fi
  if [[ "${n}" -gt 16 ]]; then n=16; fi
  if [[ "${n}" -gt "${cores}" && "${cores}" -ge 1 ]]; then n="${cores}"; fi
  echo "${n}"
}

xn_cap_to_arch() {
  local cap="$1"
  cap="$(xn_trim "${cap}")"
  if [[ "${cap}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    return
  fi
  if [[ "${cap}" =~ ^[0-9]+a?$ ]]; then
    echo "${cap}"
    return
  fi
  echo ""
}

# Blackwell consumer builds as 120a (arch-specific SASS). nvidia-smi still says 12.0.
xn_build_arch() {
  local sm="$1"
  case "${sm}" in
    120|121) echo "120a" ;;
    *) echo "${sm}" ;;
  esac
}

xn_detect_hardware() {
  XN_DEVICE="${DEVICE:-0}"
  if ! [[ "${XN_DEVICE}" =~ ^[0-9]+$ ]]; then XN_DEVICE=0; fi

  XN_CPU_CORES=1
  if command -v nproc >/dev/null 2>&1; then
    XN_CPU_CORES="$(nproc 2>/dev/null || echo 1)"
  elif command -v getconf >/dev/null 2>&1; then
    XN_CPU_CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
  fi
  XN_CPU_CORES="$(xn_trim "${XN_CPU_CORES}")"
  if ! [[ "${XN_CPU_CORES}" =~ ^[1-9][0-9]*$ ]]; then XN_CPU_CORES=1; fi

  XN_GPU_NAME=""
  XN_GPU_CAP=""
  XN_GPU_ARCH=""
  XN_GPU_VRAM_MIB=0
  XN_GPU_COUNT=0
  XN_GPU_LIST=""
  XN_GPU_FAMILY=""
  XN_ARCH_LIST=""
  XN_DETECT_OK=0
  XN_NEED_CUDA13=0
  XN_UNSUPPORTED=0
  XN_UNSUPPORTED_REASON=""

  local names caps mems
  names=()
  caps=()
  mems=()
  if command -v nvidia-smi >/dev/null 2>&1; then
    local line name cap mem
    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      name="$(printf '%s' "${line}" | cut -d, -f1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
      cap="$(printf '%s' "${line}" | cut -d, -f2 | tr -d '[:space:]')"
      mem="$(printf '%s' "${line}" | cut -d, -f3 | tr -dc '0-9')"
      [[ -z "${mem}" ]] && mem=0
      names+=("${name}")
      caps+=("${cap}")
      mems+=("${mem}")
    done < <(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
  fi

  XN_GPU_COUNT="${#names[@]}"
  if [[ "${XN_GPU_COUNT}" -gt 0 ]]; then
    local i arch build uniq seen
    seen=" "
    for i in "${!names[@]}"; do
      arch="$(xn_cap_to_arch "${caps[$i]}")"
      build="$(xn_build_arch "${arch}")"
      XN_GPU_LIST+="  [$i] ${names[$i]}  sm_${arch:-?}  ${mems[$i]} MiB"$'\n'
      if [[ -n "${build}" && "${seen}" != *" ${build} "* ]]; then
        if [[ -n "${XN_ARCH_LIST}" ]]; then XN_ARCH_LIST+=";"; fi
        XN_ARCH_LIST+="${build}"
        seen+="${build} "
      fi
    done
    if [[ "${XN_DEVICE}" -ge "${XN_GPU_COUNT}" ]]; then
      XN_DEVICE=0
    fi
    XN_GPU_NAME="${names[$XN_DEVICE]}"
    XN_GPU_CAP="${caps[$XN_DEVICE]}"
    XN_GPU_VRAM_MIB="${mems[$XN_DEVICE]}"
    XN_GPU_ARCH="$(xn_cap_to_arch "${XN_GPU_CAP}")"
    XN_GPU_FAMILY="$(xn_family_from_arch "${XN_GPU_ARCH}")"
    XN_DETECT_OK=1
  fi

  if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    XN_BUILD_ARCH="${CMAKE_CUDA_ARCHITECTURES}"
  elif [[ -n "${XN_ARCH_LIST}" ]]; then
    # All unique SMs in this box (mixed 4090+5090 → 89;120a).
    XN_BUILD_ARCH="${XN_ARCH_LIST}"
  elif [[ -n "${XN_GPU_ARCH}" ]]; then
    XN_BUILD_ARCH="$(xn_build_arch "${XN_GPU_ARCH}")"
  else
    # No nvidia-smi: fat cubin so a later box can still run.
    XN_BUILD_ARCH="75;86;89;90;120a"
  fi

  XN_NEED_CUDA13=0
  case ";${XN_ARCH_LIST};${XN_GPU_ARCH};${XN_BUILD_ARCH};" in
    *";120;"*|*";120a;"*|*";121;"*|*";100;"*|*";101;"*|*";103;"*) XN_NEED_CUDA13=1 ;;
  esac

  case "${XN_GPU_ARCH}" in
    61|60|62|52|53|50|70|72)
      XN_UNSUPPORTED=1
      XN_UNSUPPORTED_REASON="sm_${XN_GPU_ARCH} (${XN_GPU_FAMILY}) needs CUDA 12.x and is not a target of this miner (Turing sm_75 or newer)."
      ;;
  esac

  XN_SUGGESTED_LANES="$(xn_suggested_lanes "${XN_GPU_VRAM_MIB}")"
  XN_SUGGESTED_KEYGEN="$(xn_suggested_keygen "${XN_CPU_CORES}")"

  # m=100 ≈ 102.5 KiB/hash. 80% VRAM minus ~256 MiB overhead, split across lanes.
  local budget=$(( XN_GPU_VRAM_MIB * 80 / 100 ))
  if [[ "${budget}" -gt 256 ]]; then budget=$((budget - 256)); else budget=0; fi
  local lanes="${XN_SUGGESTED_LANES}"
  if [[ "${lanes}" -lt 1 ]]; then lanes=1; fi
  local per=$((budget / lanes))
  if [[ "${per}" -gt 0 ]]; then
    XN_SUGGESTED_BATCH=$(( per * 1024 / 103 ))
  else
    XN_SUGGESTED_BATCH=0
  fi
}

xn_hardware_report() {
  xn_detect_hardware
  echo "Hardware detect"
  echo "  CPU cores:     ${XN_CPU_CORES}"
  echo "  CUDA devices:  ${XN_GPU_COUNT}"
  if [[ -n "${XN_GPU_LIST}" ]]; then
    printf '%s' "${XN_GPU_LIST}"
  else
    echo "  (nvidia-smi not available — build will use a multi-arch cubin)"
  fi
  echo "  Using device:  ${XN_DEVICE}  ${XN_GPU_NAME:-unknown}  sm_${XN_GPU_ARCH:-?}  ${XN_GPU_FAMILY:-?}  ${XN_GPU_VRAM_MIB} MiB"
  echo "  Build arch:    ${XN_BUILD_ARCH}"
  echo "  Auto lanes:    ${XN_SUGGESTED_LANES}  (fixed: 128GB→32, 64→16, 32→8, 16→4, 8→2, 4–6GB→1)"
  echo "  Auto batch:    ~${XN_SUGGESTED_BATCH} hashes/lane at m=100 (fills ~80% VRAM)"
  echo "  Auto keygen:   ${XN_SUGGESTED_KEYGEN} threads (from CPU cores)"
  if [[ "${XN_NEED_CUDA13}" == "1" ]]; then
    echo "  Compiler:      CUDA 13 preferred (Blackwell)"
  else
    echo "  Compiler:      existing nvcc (CUDA 12.x or 13.x)"
  fi
  if [[ "${XN_UNSUPPORTED}" == "1" ]]; then
    echo "  WARNING:       ${XN_UNSUPPORTED_REASON}"
  fi
}

# Sourced: detect only. Executed: print report.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  xn_hardware_report
else
  xn_detect_hardware
fi
