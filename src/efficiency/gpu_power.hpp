#pragma once

#include "monitoring/nvml_monitor.hpp"
#include "util/logger.hpp"

#include <optional>

namespace xn {

class GpuPowerBooster {
public:
    GpuPowerBooster(NvmlMonitor& monitor, int target_pct, int warn_temp_c, int max_temp_c,
                    SessionLogger* logger, bool windows_performance_mode, int min_pct,
                    bool difficulty_power_enabled, int reference_difficulty,
                    double full_derate_ratio, int warn_mem_temp_c = 76, int max_mem_temp_c = 80,
                    bool use_memory_junction = true);

    void apply();
    void tick(const GpuSnapshot* snap);
    void set_difficulty(int difficulty);
    void restore();

private:
    int compute_effective_target_pct() const;
    void maybe_set_windows_high_performance(bool enable);

    NvmlMonitor& monitor_;
    SessionLogger* logger_ = nullptr;
    int target_pct_ = 100;
    int min_pct_ = 75;
    int warn_temp_c_ = 72;
    int max_temp_c_ = 75;
    int warn_mem_temp_c_ = 76;
    int max_mem_temp_c_ = 80;
    bool use_memory_junction_ = true;
    bool windows_performance_mode_ = true;
    bool difficulty_power_enabled_ = true;
    int reference_difficulty_ = 1100;
    double full_derate_ratio_ = 2.0;
    int difficulty_ = 1100;
    int effective_target_pct_ = 100;
    std::optional<unsigned int> original_limit_mw_;
    unsigned int min_limit_mw_ = 0;
    unsigned int max_limit_mw_ = 0;
    std::optional<unsigned int> current_limit_mw_;
    bool applied_ = false;
    bool apply_fail_logged_ = false;
};

}  // namespace xn
