cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0057 NEW)
project(xnminer_cuda LANGUAGES CXX CUDA)

if(WIN32)
    message(FATAL_ERROR "This tree is Linux-only. Build and run it on Linux.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default fat binary covers Turing through Blackwell. build.sh overrides this
# with the compute capability of the GPU actually in the machine (faster, native SASS).
# Blackwell consumer (RTX 50xx) builds as 120a (arch-specific SASS) and needs
# nvcc 12.8+ (CUDA 13 preferred). If cmake is run by hand with no -D: native on 3.24+.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75;86;89;90;120a")
    endif()
endif()

set(VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

add_executable(xnminer
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/config/settings.cpp
    ${SRC_DIR}/util/http.cpp
    ${SRC_DIR}/util/logger.cpp
    ${SRC_DIR}/util/paths.cpp
    ${SRC_DIR}/util/sha256.cpp
    ${SRC_DIR}/util/cpu.cpp
    ${SRC_DIR}/mining/argon2_encode.cpp
    ${SRC_DIR}/mining/block_types.cpp
    ${SRC_DIR}/mining/cuda_engine.cpp
    ${SRC_DIR}/mining/vram_batch.cpp
    ${SRC_DIR}/efficiency/vram_policy.cpp
    ${SRC_DIR}/efficiency/thermal_policy.cpp
    ${SRC_DIR}/efficiency/gpu_power.cpp
    ${SRC_DIR}/network/difficulty.cpp
    ${SRC_DIR}/network/lastblock.cpp
    ${SRC_DIR}/network/submitter.cpp
    ${SRC_DIR}/network/bag_forward.cpp
    ${SRC_DIR}/network/updater.cpp
    ${SRC_DIR}/queue/store.cpp
    ${SRC_DIR}/queue/policy.cpp
    ${SRC_DIR}/monitoring/nvml_monitor.cpp
    ${SRC_DIR}/monitoring/nvapi_temps.cpp
    ${SRC_DIR}/monitoring/metrics.cpp
    ${SRC_DIR}/monitoring/dashboard.cpp
    ${SRC_DIR}/monitoring/woodyminer.cpp
    ${SRC_DIR}/monitoring/wallet.cpp
    ${SRC_DIR}/monitoring/xenblockscan.cpp
    ${SRC_DIR}/monitoring/timelapse.cpp
    ${SRC_DIR}/monitoring/local_stats.cpp
    ${SRC_DIR}/app/supervisor.cpp
    ${SRC_DIR}/app/instance_lock.cpp
    ${VENDOR_DIR}/argon2params.cpp
    ${VENDOR_DIR}/blake2b.cpp
    ${VENDOR_DIR}/CudaBackend.cpp
    ${VENDOR_DIR}/CudaDevice.cpp
    ${VENDOR_DIR}/kernelrunner.cu
    ${VENDOR_DIR}/hashapi/CudaHashBackend.cpp
    ${VENDOR_DIR}/hashapi/KeygenPool.cpp
    ${VENDOR_DIR}/hashapi/HashApiEncoding.cpp
    ${VENDOR_DIR}/hashapi/HashApiMatching.cpp
    ${VENDOR_DIR}/hashapi/HashApiTuning.cpp
    ${VENDOR_DIR}/hashapi/HashApiValidation.cpp
)

target_include_directories(xnminer PRIVATE
    ${SRC_DIR}
    ${VENDOR_DIR}
    ${VENDOR_DIR}/hashapi
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)
# Static cudart: the fat binary must run on HiveOS (no CUDA toolkit installed).
target_link_libraries(xnminer PRIVATE CUDA::cudart_static Threads::Threads CURL::libcurl ${CMAKE_DL_LIBS})
target_compile_definitions(xnminer PRIVATE _GNU_SOURCE)

target_compile_options(xnminer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-O3,-dlcm=ca>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Kernel hint only when this cubin includes Blackwell SASS.
string(REPLACE "," ";" _xn_cuda_arches "${CMAKE_CUDA_ARCHITECTURES}")
if("120" IN_LIST _xn_cuda_arches OR "120a" IN_LIST _xn_cuda_arches)
    target_compile_definitions(xnminer PRIVATE XN_BLACKWELL_BUILD=1)
endif()

set_target_properties(xnminer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION OFF
)
