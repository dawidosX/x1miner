#include "efficiency/gpu_power.hpp"

#include "efficiency/thermal_policy.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <cctype>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#endif

#include <algorithm>
#include <cmath>
#include <string>

namespace xn {

GpuPowerBooster::GpuPowerBooster(NvmlMonitor& monitor, int target_pct, int warn_temp_c,
                                 int max_temp_c, SessionLogger* logger,
                                 bool windows_performance_mode, int min_pct,
                                 bool difficulty_power_enabled, int reference_difficulty,
                                 double full_derate_ratio, int warn_mem_temp_c, int max_mem_temp_c,
                                 bool use_memory_junction)
    : monitor_(monitor),
      logger_(logger),
      warn_temp_c_(warn_temp_c),
      max_temp_c_(max_temp_c),
      warn_mem_temp_c_(warn_mem_temp_c),
      max_mem_temp_c_(max_mem_temp_c),
      use_memory_junction_(use_memory_junction),
      windows_performance_mode_(windows_performance_mode),
      difficulty_power_enabled_(difficulty_power_enabled),
      reference_difficulty_(std::max(1, reference_difficulty)),
      full_derate_ratio_(std::max(1.01, full_derate_ratio)),
      difficulty_(std::max(1, reference_difficulty)) {
    normalize_power_range(target_pct, min_pct);
    target_pct_ = target_pct;
    min_pct_ = min_pct;
    effective_target_pct_ = target_pct_;
}

int GpuPowerBooster::compute_effective_target_pct() const {
    if (!difficulty_power_enabled_) return target_pct_;
    return difficulty_power_target_pct(target_pct_, difficulty_, reference_difficulty_, min_pct_,
                                       full_derate_ratio_);
}

void GpuPowerBooster::set_difficulty(int difficulty) {
    difficulty_ = std::max(1, difficulty);
    int new_pct = compute_effective_target_pct();
    if (new_pct != effective_target_pct_) {
        if (logger_) {
            logger_->info("GPU power target " + std::to_string(effective_target_pct_) + "% -> " +
                          std::to_string(new_pct) + "% (difficulty=" + std::to_string(difficulty_) +
                          ")");
        }
        effective_target_pct_ = new_pct;
    }
}

void GpuPowerBooster::maybe_set_windows_high_performance(bool enable) {
    if (!windows_performance_mode_ || !enable) return;
#ifdef _WIN32
    system("powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1");
#else
    (void)std::system("nvidia-smi -pm 1 >/dev/null 2>&1");
    DIR* dir = opendir("/sys/devices/system/cpu");
    if (dir) {
        while (dirent* ent = readdir(dir)) {
            const std::string name = ent->d_name;
            if (name.rfind("cpu", 0) != 0) continue;
            if (name.size() < 4 || !std::isdigit(static_cast<unsigned char>(name[3]))) continue;
            const std::string path =
                "/sys/devices/system/cpu/" + name + "/cpufreq/scaling_governor";
            std::ofstream out(path);
            if (out) out << "performance\n";
        }
        closedir(dir);
    }
#endif
}

void GpuPowerBooster::apply() {
    if (!monitor_.available()) return;
    auto limits = monitor_.get_power_limits_mw();
    if (!limits) {
        if (logger_) logger_->warn("NVML power limits unavailable");
        return;
    }
    original_limit_mw_ = limits->current_mw;
    min_limit_mw_ = limits->min_mw;
    max_limit_mw_ = limits->max_mw;
    maybe_set_windows_high_performance(true);

    unsigned int desired =
        static_cast<unsigned int>(std::llround(max_limit_mw_ * (effective_target_pct_ / 100.0)));
    desired = std::clamp(desired, min_limit_mw_, max_limit_mw_);
    if (monitor_.set_power_limit_mw(desired)) {
        current_limit_mw_ = desired;
        applied_ = true;
        if (logger_) {
            logger_->info("GPU power limit set to " + std::to_string(desired / 1000) + "W (" +
                          std::to_string(effective_target_pct_) + "% of max)");
        }
    } else {
#ifndef _WIN32
        const unsigned watts = desired / 1000;
        const std::string cmd = "nvidia-smi -i " + std::to_string(0) + " -pl " +
                                std::to_string(watts) + " >/dev/null 2>&1";
        if (std::system(cmd.c_str()) == 0) {
            current_limit_mw_ = desired;
            applied_ = true;
            if (logger_) {
                logger_->info("GPU power limit set via nvidia-smi to " + std::to_string(watts) +
                              "W (" + std::to_string(effective_target_pct_) + "% of max)");
            }
        } else
#endif
            if (logger_ && !apply_fail_logged_) {
            apply_fail_logged_ = true;
#ifdef _WIN32
            logger_->warn(
                "Failed to set GPU power limit (try Administrator) — thermal idle will shed heat instead");
#else
            logger_->warn(
                "Failed to set GPU power limit (NVML + nvidia-smi) — check persistence mode / root");
#endif
        }
    }
}

void GpuPowerBooster::tick(const GpuSnapshot* snap) {
    if (!monitor_.available()) return;
    if (!applied_) {
        apply();
        if (!applied_) return;
    }
    int pct = effective_target_pct_;
    if (snap) {
        double heat = 0.0;  // 0 = cool, 1 = at/over max
        if (snap->temperature_c >= max_temp_c_) {
            heat = 1.0;
        } else if (snap->temperature_c >= warn_temp_c_) {
            heat = static_cast<double>(snap->temperature_c - warn_temp_c_) /
                   std::max(1, max_temp_c_ - warn_temp_c_);
        }
        if (use_memory_junction_ && snap->has_memory_junction()) {
            double mem_heat = 0.0;
            if (snap->memory_junction_c >= max_mem_temp_c_) {
                mem_heat = 1.0;
            } else if (snap->memory_junction_c >= warn_mem_temp_c_) {
                mem_heat = static_cast<double>(snap->memory_junction_c - warn_mem_temp_c_) /
                           std::max(1, max_mem_temp_c_ - warn_mem_temp_c_);
            }
            heat = std::max(heat, mem_heat);
        }
        if (heat >= 1.0) {
            pct = min_pct_;
        } else if (heat > 0.0) {
            pct = static_cast<int>(
                std::llround(effective_target_pct_ - heat * (effective_target_pct_ - min_pct_)));
        }
    }
    unsigned int desired =
        static_cast<unsigned int>(std::llround(max_limit_mw_ * (pct / 100.0)));
    desired = std::clamp(desired, min_limit_mw_, max_limit_mw_);
    if (!current_limit_mw_ || *current_limit_mw_ != desired) {
        if (monitor_.set_power_limit_mw(desired)) current_limit_mw_ = desired;
    }
}

void GpuPowerBooster::restore() {
    if (original_limit_mw_ && monitor_.available()) {
        monitor_.set_power_limit_mw(*original_limit_mw_);
    }
    applied_ = false;
}

}  // namespace xn
