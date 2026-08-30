#pragma once

namespace xn {

double clamp_float(double v, double lo, double hi);
double thermal_batch_scale(int temperature_c, int warn_temp_c, int max_temp_c,
                           double min_scale = 0.70);
// Worse (lower) of GPU-die and optional memory-junction scales.
double thermal_batch_scale_sensors(int gpu_c, int warn_gpu_c, int max_gpu_c, int mem_c,
                                   int warn_mem_c, int max_mem_c, bool use_memory_junction,
                                   double min_scale = 0.70);

// Closed-loop hunt: step batch by a fixed count, then idle the GPU between
// waves if heat is still climbing (batch size alone does not drop duty cycle).
struct ThermalHuntState {
    int batch = 0;
    double last_adjust_s = 0;
    int last_temp_c = 0;
    int consecutive_cool = 0;
    int over_floor_streak = 0;
    int idle_ms = 0;
    /// Rising low watermark. Cuts never go to or below this.
    int floor = 0;
    /// Last job count while junction was at or under the hold.
    int last_ok = 0;
};

struct ThermalHuntResult {
    int batch = 0;
    double scale = 1.0;
    const char* action = "hold";  // hold | cut | raise
    int control_c = 0;
    const char* sensor = "gpu";
    int idle_ms = 0;
    int lane_delta = 0;  // -1 drop one lane, +1 restore one
};

ThermalHuntResult hunt_thermal_scale(ThermalHuntState& st, int gpu_c, int mem_c, bool use_mem,
                                     int target_c, int cap_c, int gpu_cap_c, int planned_batch,
                                     int step, int min_batch, double now_s,
                                     double cut_settle_s = 60.0, double raise_settle_s = 75.0);

int apply_batch_scale(int batch, double scale);
int difficulty_power_target_pct(int target_pct, int difficulty, int reference_difficulty,
                                int min_pct = 75, double full_derate_ratio = 2.0);
void normalize_power_range(int& target_pct, int& min_pct);

}  // namespace xn
