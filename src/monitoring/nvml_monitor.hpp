#pragma once

#include "common.hpp"

#include <optional>
#include <string>

namespace xn {

struct PowerLimits {
    unsigned int current_mw = 0;
    unsigned int min_mw = 0;
    unsigned int max_mw = 0;
};

class NvmlMonitor {
public:
    explicit NvmlMonitor(int device_index = 0);
    ~NvmlMonitor();

    bool available() const { return ready_; }
    std::optional<GpuSnapshot> snapshot() const;
    std::optional<PowerLimits> get_power_limits_mw() const;
    bool set_power_limit_mw(unsigned int limit_mw);
    void shutdown();

private:
    int device_index_ = 0;
    bool ready_ = false;
    void* handle_ = nullptr;  // nvmlDevice_t
    bool owns_init_ = false;
};

}  // namespace xn
