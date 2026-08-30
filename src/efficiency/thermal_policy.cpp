#include "efficiency/thermal_policy.hpp"

#include <algorithm>
#include <cmath>

namespace xn {

double clamp_float(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

double thermal_batch_scale(int temperature_c, int warn_temp_c, int max_temp_c, double min_scale) {
    min_scale = clamp_float(min_scale, 0.50, 1.0);
    if (max_temp_c <= warn_temp_c) return min_scale;
    if (temperature_c <= warn_temp_c) return 1.0;
    if (temperature_c >= max_temp_c) return min_scale;
    double t = static_cast<double>(temperature_c - warn_temp_c) /
               static_cast<double>(max_temp_c - warn_temp_c);
    return clamp_float(1.0 - t * (1.0 - min_scale), min_scale, 1.0);
}

double thermal_batch_scale_sensors(int gpu_c, int warn_gpu_c, int max_gpu_c, int mem_c,
                                   int warn_mem_c, int max_mem_c, bool use_memory_junction,
                                   double min_scale) {
    double scale = thermal_batch_scale(gpu_c, warn_gpu_c, max_gpu_c, min_scale);
    if (use_memory_junction && mem_c > 0) {
        scale = std::min(scale, thermal_batch_scale(mem_c, warn_mem_c, max_mem_c, min_scale));
    }
    return scale;
}

ThermalHuntResult hunt_thermal_scale(ThermalHuntState& st, int gpu_c, int mem_c, bool use_mem,
                                     int target_c, int cap_c, int gpu_cap_c, int planned_batch,
                                     int step, int min_batch, double now_s, double cut_settle_s,
                                     double raise_settle_s) {
    // Fast cruise: 5s is enough to see junction move. The old 20s floor let
    // the card run to 84C before the first cut.
    cut_settle_s = std::max(5.0, cut_settle_s);
    raise_settle_s = std::max(cut_settle_s + 5.0, raise_settle_s);
    if (step < 1) step = 1000;
    if (planned_batch < 1) planned_batch = 1;
    if (min_batch < step) min_batch = step;
    if (min_batch > planned_batch) min_batch = planned_batch;
    if (st.batch <= 0) st.batch = planned_batch;
    if (st.batch > planned_batch) st.batch = planned_batch;
    if (st.batch < min_batch) st.batch = min_batch;

    ThermalHuntResult r;
    const bool junction = use_mem && mem_c > 0;
    r.sensor = junction ? "mem" : "gpu";
    r.control_c = junction ? mem_c : gpu_c;
    if (target_c < 40) target_c = 40;
    if (cap_c <= target_c) cap_c = target_c + 3;

    if (gpu_cap_c > 0 && gpu_c >= gpu_cap_c && gpu_c > r.control_c) {
        r.control_c = gpu_c;
        r.sensor = "gpu";
    }

    int temp = r.control_c;
    // A 12C+ drop in one sample is a glitch (NVAPI miss). Ignore it.
    if (st.last_temp_c >= target_c && temp > 0 && st.last_temp_c - temp >= 12) {
        temp = st.last_temp_c;
        r.control_c = temp;
    }

    // First reading: start the observe window. Do not move yet.
    if (st.last_adjust_s <= 0) {
        st.last_adjust_s = now_s;
        st.last_temp_c = temp;
        if (st.floor <= 0) st.floor = st.batch;
        if (st.last_ok <= 0) st.last_ok = st.batch;
        r.action = "hold";
        r.batch = st.batch;
        r.idle_ms = 0;
        r.scale = static_cast<double>(st.batch) / static_cast<double>(planned_batch);
        return r;
    }

    if (st.floor <= 0) st.floor = st.batch;
    if (st.last_ok <= 0) st.last_ok = st.batch;
    if (st.floor < min_batch) st.floor = min_batch;

    const double since = now_s - st.last_adjust_s;
    r.action = "hold";
    r.lane_delta = 0;

    int want_idle = 0;
    if (temp >= target_c) {
        want_idle = std::min(80, 12 * (temp - target_c + 1));
    }
    st.idle_ms = want_idle;

    const int dT = (st.last_temp_c > 0) ? (temp - st.last_temp_c) : 0;
    const bool cool = (temp <= target_c && dT <= 0);
    const bool hot = (temp >= cap_c || temp >= target_c + 2 ||
                      (temp >= target_c && dT > 0));

    auto apply = [&](int next, const char* action) {
        if (since < cut_settle_s) return;
        if (next > planned_batch) next = planned_batch;
        if (next < min_batch) next = min_batch;
        if (next == st.batch) return;
        st.batch = next;
        st.last_adjust_s = now_s;
        r.action = action;
    };

    if (cool) {
        st.last_ok = st.batch;
        st.consecutive_cool++;
        apply(st.batch + step, "raise");
    } else if (hot) {
        st.consecutive_cool = 0;
        // Drop to last good count, but always ABOVE the previous low.
        int drop = st.last_ok;
        if (drop <= st.floor) drop = st.floor + step;
        if (drop >= st.batch) drop = st.batch - step;
        if (drop <= st.floor) drop = st.floor + step;
        if (drop < min_batch) drop = min_batch;
        if (drop > st.batch) drop = st.batch;
        if (drop != st.batch) {
            apply(drop, "cut");
            if (st.batch > st.floor) st.floor = st.batch;
        }
    } else {
        st.consecutive_cool = 0;
    }

    st.last_temp_c = temp;
    r.batch = st.batch;
    r.idle_ms = st.idle_ms;
    r.scale = static_cast<double>(st.batch) / static_cast<double>(planned_batch);
    return r;
}

int apply_batch_scale(int batch, double scale) {
    if (batch <= 0) return 0;
    int scaled = static_cast<int>(std::llround(batch * clamp_float(scale, 0.0, 1.0)));
    return std::max(1, scaled);
}

int difficulty_power_target_pct(int target_pct, int difficulty, int reference_difficulty,
                                int min_pct, double full_derate_ratio) {
    normalize_power_range(target_pct, min_pct);
    if (reference_difficulty <= 0 || difficulty <= reference_difficulty) return target_pct;
    double ratio = static_cast<double>(difficulty) / static_cast<double>(reference_difficulty);
    double span = std::max(1.01, full_derate_ratio) - 1.0;
    double t = clamp_float((ratio - 1.0) / span, 0.0, 1.0);
    return static_cast<int>(std::llround(target_pct - t * (target_pct - min_pct)));
}

void normalize_power_range(int& target_pct, int& min_pct) {
    target_pct = std::clamp(target_pct, 50, 100);
    min_pct = std::clamp(min_pct, 50, target_pct);
}

}  // namespace xn
