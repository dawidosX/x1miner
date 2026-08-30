#include "efficiency/vram_policy.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace xn {

std::string VramCaps::summary() const {
    std::ostringstream oss;
    oss << "VRAM caps on " << total_mib << "MiB GPU: target<=" << target_mib << "MiB ("
        << static_cast<int>(target_pct) << "%) desktop>=" << headroom_mib << "MiB free ("
        << static_cast<int>(headroom_pct) << "%) emergency>=" << emergency_mib << "MiB used ("
        << static_cast<int>(emergency_pct) << "%) min_free=" << min_headroom_mib
        << "MiB cuda_overhead=" << runtime_overhead_mib << "MiB";
    return oss.str();
}

static double clamp_pct(double v, double lo = 0.0, double hi = 100.0) {
    return std::max(lo, std::min(hi, v));
}

VramCaps resolve_vram_caps(int total_mib, double target_pct, double desktop_headroom_pct,
                           double emergency_vram_pct, double min_headroom_pct,
                           double runtime_overhead_pct, int min_headroom_floor_mib,
                           int overhead_floor_mib, int target_mib_override,
                           int headroom_mib_override, int emergency_mib_override,
                           int min_headroom_mib_override, int runtime_overhead_mib_override) {
    int total = std::max(1, total_mib);
    double t_pct = clamp_pct(target_pct);
    double h_pct = clamp_pct(desktop_headroom_pct);
    double e_pct = clamp_pct(emergency_vram_pct);
    double m_pct = clamp_pct(min_headroom_pct);
    double o_pct = clamp_pct(runtime_overhead_pct);

    int target = target_mib_override > 0
                     ? target_mib_override
                     : static_cast<int>(std::llround(total * t_pct / 100.0));
    int headroom = headroom_mib_override > 0
                       ? headroom_mib_override
                       : static_cast<int>(std::llround(total * h_pct / 100.0));
    int emergency = emergency_mib_override > 0
                        ? emergency_mib_override
                        : static_cast<int>(std::llround(total * e_pct / 100.0));
    int min_head = min_headroom_mib_override > 0
                       ? min_headroom_mib_override
                       : std::max(min_headroom_floor_mib,
                                  static_cast<int>(std::llround(total * m_pct / 100.0)));
    int overhead = runtime_overhead_mib_override > 0
                       ? runtime_overhead_mib_override
                       : std::max(overhead_floor_mib,
                                  static_cast<int>(std::llround(total * o_pct / 100.0)));

    target = std::clamp(target, 1, total);
    headroom = std::clamp(headroom, 0, total);
    emergency = std::clamp(emergency, target, total);
    min_head = std::clamp(min_head, 0, total);
    overhead = std::clamp(overhead, 0, total);

    VramCaps c;
    c.total_mib = total;
    c.target_mib = target;
    c.headroom_mib = headroom;
    c.emergency_mib = emergency;
    c.min_headroom_mib = min_head;
    c.runtime_overhead_mib = overhead;
    c.target_pct = t_pct;
    c.headroom_pct = h_pct;
    c.emergency_pct = e_pct;
    c.min_headroom_pct = m_pct;
    c.overhead_pct = o_pct;
    return c;
}

}  // namespace xn
