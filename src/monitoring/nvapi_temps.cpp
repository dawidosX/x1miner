#include "monitoring/nvapi_temps.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdint>

namespace xn {
namespace {

using NvStatus = int;
using NvGpu = void*;
constexpr int kNvOk = 0;

#pragma pack(push, 8)
struct NvThermalSensors {
    unsigned int version = 0;
    unsigned int mask = 0;
    int reserved[8] = {};
    int temperatures[32] = {};
};
#pragma pack(pop)

using QiFn = void* (*)(unsigned int);
using InitFn = NvStatus (*)();
using EnumFn = NvStatus (*)(NvGpu*, int*);
using ThermFn = NvStatus (*)(NvGpu, NvThermalSensors*);

struct NvapiTemps {
    bool tried = false;
    bool ok = false;
    NvGpu gpu = nullptr;
    unsigned int mask = 0;
    ThermFn get_therm = nullptr;
};

NvapiTemps& state() {
    static NvapiTemps s;
    return s;
}

int decode_temp(int raw) {
    // Values are 8.8 fixed point. 255.00 C (65280) is an empty slot.
    if (raw <= 0) return 0;
    int c = raw / 256;
    if (c <= 0 || c >= 200) return 0;
    return c;
}

void init(int device_index) {
    auto& s = state();
    if (s.tried) return;
    s.tried = true;
#ifdef _WIN32
    HMODULE lib = LoadLibraryA("nvapi64.dll");
    if (!lib) return;
    auto qi = reinterpret_cast<QiFn>(GetProcAddress(lib, "nvapi_QueryInterface"));
    if (!qi) return;
    auto init_fn = reinterpret_cast<InitFn>(qi(0x0150E828u));
    auto enum_fn = reinterpret_cast<EnumFn>(qi(0xE5AC921Fu));
    auto therm_fn = reinterpret_cast<ThermFn>(qi(0x65FE3AADu));
    if (!init_fn || !enum_fn || !therm_fn) return;
    if (init_fn() != kNvOk) return;
    NvGpu gpus[64] = {};
    int n = 0;
    if (enum_fn(gpus, &n) != kNvOk || n <= 0) return;
    int idx = device_index;
    if (idx < 0 || idx >= n) idx = 0;
    s.gpu = gpus[idx];
    s.get_therm = therm_fn;

    unsigned int mask = 0;
    for (int bit = 0; bit < 32; ++bit) {
        NvThermalSensors ts{};
        ts.version = static_cast<unsigned int>(sizeof(NvThermalSensors) | (2u << 16));
        ts.mask = 1u << bit;
        if (therm_fn(s.gpu, &ts) == kNvOk) {
            mask = ts.mask;
            continue;
        }
        mask = (1u << bit) - 1u;
        break;
    }
    s.mask = mask ? mask : 0x00FFFFFFu;
    s.ok = true;
#else
    (void)device_index;
#endif
}

}  // namespace

std::optional<int> read_nvapi_memory_junction_c(int device_index) {
    init(device_index);
    auto& s = state();
    if (!s.ok || !s.get_therm || !s.gpu) return std::nullopt;

    NvThermalSensors ts{};
    ts.version = static_cast<unsigned int>(sizeof(NvThermalSensors) | (2u << 16));
    ts.mask = s.mask;
    if (s.get_therm(s.gpu, &ts) != kNvOk) return std::nullopt;

    // LibreHardwareMonitor RTX 50xx mapping: [1]=core, [2]=memory junction.
    int mem = decode_temp(ts.temperatures[2]);
    if (mem > 0) return mem;
    // Older cards: junction often lives at [9] (Ampere) or [7] (Ada).
    for (int idx : {9, 7, 3}) {
        mem = decode_temp(ts.temperatures[idx]);
        if (mem > 0) return mem;
    }
    return std::nullopt;
}

}  // namespace xn
