#pragma once

#include <optional>

namespace xn {

// Optional Windows NvAPI memory-junction read. Unused on Linux (NVML path).
std::optional<int> read_nvapi_memory_junction_c(int device_index);

}  // namespace xn
