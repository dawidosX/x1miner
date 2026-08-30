#include "monitoring/nvml_monitor.hpp"
#include "monitoring/nvapi_temps.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>

// Dynamic NVML loading — never hard-depends on a static import.
namespace {

#ifdef _WIN32
using LibHandle = HMODULE;
LibHandle load_lib(const char* name) { return LoadLibraryA(name); }
void* load_sym(LibHandle h, const char* name) {
    return h ? reinterpret_cast<void*>(GetProcAddress(h, name)) : nullptr;
}
void close_lib(LibHandle h) {
    if (h) FreeLibrary(h);
}
#else
using LibHandle = void*;
LibHandle load_lib(const char* name) { return dlopen(name, RTLD_LAZY | RTLD_LOCAL); }
void* load_sym(LibHandle h, const char* name) { return h ? dlsym(h, name) : nullptr; }
void close_lib(LibHandle h) {
    if (h) dlclose(h);
}
#endif

using nvmlReturn_t = int;
constexpr int NVML_SUCCESS = 0;
constexpr int NVML_TEMPERATURE_GPU = 0;
// nvml.h: NVML_FI_DEV_MEMORY_TEMP — GDDR memory junction (°C)
constexpr unsigned int NVML_FI_DEV_MEMORY_TEMP = 82;
constexpr int NVML_VALUE_TYPE_DOUBLE = 0;
constexpr int NVML_VALUE_TYPE_UNSIGNED_INT = 1;
constexpr int NVML_VALUE_TYPE_UNSIGNED_LONG = 2;
constexpr int NVML_VALUE_TYPE_UNSIGNED_LONG_LONG = 3;
constexpr int NVML_VALUE_TYPE_SIGNED_LONG_LONG = 4;
constexpr int NVML_VALUE_TYPE_SIGNED_INT = 5;
constexpr int NVML_VALUE_TYPE_UNSIGNED_SHORT = 6;

union NvmlValue {
    double dVal;
    int siVal;
    unsigned int uiVal;
    unsigned long ulVal;
    unsigned long long ullVal;
    long long sllVal;
    unsigned short usVal;
};

// Must match nvmlFieldValue_t in nvml.h (Windows x64).
struct NvmlFieldValue {
    unsigned int fieldId = 0;
    unsigned int scopeId = 0;
    long long timestamp = 0;
    long long latencyUsec = 0;
    int valueType = 0;
    int nvmlReturn = 0;
    NvmlValue value{};
};

int field_value_as_int(const NvmlFieldValue& fv) {
    if (fv.nvmlReturn != NVML_SUCCESS) return 0;
    switch (fv.valueType) {
        case NVML_VALUE_TYPE_DOUBLE:
            return static_cast<int>(fv.value.dVal);
        case NVML_VALUE_TYPE_UNSIGNED_INT:
            return static_cast<int>(fv.value.uiVal);
        case NVML_VALUE_TYPE_UNSIGNED_LONG:
            return static_cast<int>(fv.value.ulVal);
        case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
            return static_cast<int>(fv.value.ullVal);
        case NVML_VALUE_TYPE_SIGNED_LONG_LONG:
            return static_cast<int>(fv.value.sllVal);
        case NVML_VALUE_TYPE_SIGNED_INT:
            return fv.value.siVal;
        case NVML_VALUE_TYPE_UNSIGNED_SHORT:
            return static_cast<int>(fv.value.usVal);
        default:
            return static_cast<int>(fv.value.uiVal);
    }
}

struct NvmlFns {
    LibHandle lib = nullptr;
    nvmlReturn_t (*nvmlInit_v2)() = nullptr;
    nvmlReturn_t (*nvmlShutdown)() = nullptr;
    nvmlReturn_t (*nvmlDeviceGetHandleByIndex_v2)(unsigned int, void**) = nullptr;
    nvmlReturn_t (*nvmlDeviceGetName)(void*, char*, unsigned int) = nullptr;
    nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(void*, void*) = nullptr;
    nvmlReturn_t (*nvmlDeviceGetUtilizationRates)(void*, void*) = nullptr;
    nvmlReturn_t (*nvmlDeviceGetPowerUsage)(void*, unsigned int*) = nullptr;
    nvmlReturn_t (*nvmlDeviceGetTemperature)(void*, int, unsigned int*) = nullptr;
    nvmlReturn_t (*nvmlDeviceGetFieldValues)(void*, int, NvmlFieldValue*) = nullptr;
    nvmlReturn_t (*nvmlDeviceGetPowerManagementLimit)(void*, unsigned int*) = nullptr;
    nvmlReturn_t (*nvmlDeviceGetPowerManagementLimitConstraints)(void*, unsigned int*,
                                                                 unsigned int*) = nullptr;
    nvmlReturn_t (*nvmlDeviceSetPowerManagementLimit)(void*, unsigned int) = nullptr;
    bool ok = false;
};

struct NvmlMemory {
    unsigned long long total = 0;
    unsigned long long free = 0;
    unsigned long long used = 0;
};

struct NvmlUtilization {
    unsigned int gpu = 0;
    unsigned int memory = 0;
};

NvmlFns& fns() {
    static NvmlFns f;
    static bool tried = false;
    if (!tried) {
        tried = true;
#ifdef _WIN32
        f.lib = load_lib("nvml.dll");
        if (!f.lib) {
            f.lib = load_lib("C:\\Windows\\System32\\nvml.dll");
        }
#else
        static const char* kCandidates[] = {
            "libnvidia-ml.so.1",
            "libnvidia-ml.so",
            "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1",
            "/usr/lib64/libnvidia-ml.so.1",
            "/usr/lib/libnvidia-ml.so.1",
        };
        for (const char* name : kCandidates) {
            f.lib = load_lib(name);
            if (f.lib) break;
        }
#endif
        if (!f.lib) return f;
#define LOAD(name) f.name = reinterpret_cast<decltype(f.name)>(load_sym(f.lib, #name))
        LOAD(nvmlInit_v2);
        LOAD(nvmlShutdown);
        LOAD(nvmlDeviceGetHandleByIndex_v2);
        LOAD(nvmlDeviceGetName);
        LOAD(nvmlDeviceGetMemoryInfo);
        LOAD(nvmlDeviceGetUtilizationRates);
        LOAD(nvmlDeviceGetPowerUsage);
        LOAD(nvmlDeviceGetTemperature);
        LOAD(nvmlDeviceGetFieldValues);
        LOAD(nvmlDeviceGetPowerManagementLimit);
        LOAD(nvmlDeviceGetPowerManagementLimitConstraints);
        LOAD(nvmlDeviceSetPowerManagementLimit);
#undef LOAD
        f.ok = f.nvmlInit_v2 && f.nvmlDeviceGetHandleByIndex_v2 && f.nvmlDeviceGetMemoryInfo;
    }
    return f;
}

}  // namespace

namespace xn {

NvmlMonitor::NvmlMonitor(int device_index) : device_index_(device_index) {
    auto& f = fns();
    if (!f.ok) return;
    if (f.nvmlInit_v2() != NVML_SUCCESS) return;
    owns_init_ = true;
    if (f.nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned int>(device_index_), &handle_) !=
        NVML_SUCCESS) {
        return;
    }
    ready_ = true;
}

NvmlMonitor::~NvmlMonitor() { shutdown(); }

void NvmlMonitor::shutdown() {
    if (owns_init_) {
        auto& f = fns();
        if (f.nvmlShutdown) f.nvmlShutdown();
        owns_init_ = false;
    }
    ready_ = false;
    handle_ = nullptr;
}

std::optional<GpuSnapshot> NvmlMonitor::snapshot() const {
    if (!ready_) {
        // Fallback via CUDA runtime memory only.
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return std::nullopt;
        GpuSnapshot s;
        s.index = device_index_;
        s.name = "CUDA GPU";
        s.total_mib = static_cast<int>(total_b / (1024 * 1024));
        s.free_mib = static_cast<int>(free_b / (1024 * 1024));
        s.used_mib = s.total_mib - s.free_mib;
        return s;
    }
    auto& f = fns();
    NvmlMemory mem{};
    if (f.nvmlDeviceGetMemoryInfo(handle_, &mem) != NVML_SUCCESS) return std::nullopt;
    GpuSnapshot s;
    s.index = device_index_;
    char name[96] = {};
    if (f.nvmlDeviceGetName && f.nvmlDeviceGetName(handle_, name, sizeof(name)) == NVML_SUCCESS) {
        s.name = name;
    } else {
        s.name = "NVIDIA GPU";
    }
    s.total_mib = static_cast<int>(mem.total / (1024 * 1024));
    s.used_mib = static_cast<int>(mem.used / (1024 * 1024));
    s.free_mib = static_cast<int>(mem.free / (1024 * 1024));
    if (f.nvmlDeviceGetUtilizationRates) {
        NvmlUtilization u{};
        if (f.nvmlDeviceGetUtilizationRates(handle_, &u) == NVML_SUCCESS) s.util_pct = static_cast<int>(u.gpu);
    }
    if (f.nvmlDeviceGetPowerUsage) {
        unsigned int mw = 0;
        if (f.nvmlDeviceGetPowerUsage(handle_, &mw) == NVML_SUCCESS) s.power_w = mw / 1000.0;
    }
    if (f.nvmlDeviceGetTemperature) {
        unsigned int temp = 0;
        if (f.nvmlDeviceGetTemperature(handle_, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            s.temperature_c = static_cast<int>(temp);
        }
        // Sensor 1 = memory on many boards.
        unsigned int memt = 0;
        if (f.nvmlDeviceGetTemperature(handle_, 1, &memt) == NVML_SUCCESS && memt > 0 &&
            memt < 200) {
            s.memory_junction_c = static_cast<int>(memt);
        }
    }
    // Memory junction via NVML field when the driver exposes it.
    if (f.nvmlDeviceGetFieldValues) {
        NvmlFieldValue fv{};
        fv.fieldId = NVML_FI_DEV_MEMORY_TEMP;
        if (f.nvmlDeviceGetFieldValues(handle_, 1, &fv) == NVML_SUCCESS) {
            int mem = field_value_as_int(fv);
            if (mem > 0 && mem < 200) s.memory_junction_c = mem;
        }
    }
    if (s.memory_junction_c <= 0) {
        if (auto nvapi = read_nvapi_memory_junction_c(device_index_)) {
            s.memory_junction_c = *nvapi;
        }
    }
    return s;
}

std::optional<PowerLimits> NvmlMonitor::get_power_limits_mw() const {
    if (!ready_) return std::nullopt;
    auto& f = fns();
    if (!f.nvmlDeviceGetPowerManagementLimit || !f.nvmlDeviceGetPowerManagementLimitConstraints)
        return std::nullopt;
    PowerLimits pl;
    if (f.nvmlDeviceGetPowerManagementLimit(handle_, &pl.current_mw) != NVML_SUCCESS) return std::nullopt;
    if (f.nvmlDeviceGetPowerManagementLimitConstraints(handle_, &pl.min_mw, &pl.max_mw) !=
        NVML_SUCCESS)
        return std::nullopt;
    return pl;
}

bool NvmlMonitor::set_power_limit_mw(unsigned int limit_mw) {
    if (!ready_) return false;
    auto& f = fns();
    if (!f.nvmlDeviceSetPowerManagementLimit) return false;
    return f.nvmlDeviceSetPowerManagementLimit(handle_, limit_mw) == NVML_SUCCESS;
}

}  // namespace xn
