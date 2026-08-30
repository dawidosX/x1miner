#pragma once

#include <string>

namespace xn {

struct VramCaps {
    int total_mib = 0;
    int target_mib = 0;
    int headroom_mib = 0;
    int emergency_mib = 0;
    int min_headroom_mib = 0;
    int runtime_overhead_mib = 0;
    double target_pct = 0;
    double headroom_pct = 0;
    double emergency_pct = 0;
    double min_headroom_pct = 0;
    double overhead_pct = 0;

    std::string summary() const;
};

VramCaps resolve_vram_caps(
    int total_mib, double target_pct = 79.0, double desktop_headroom_pct = 16.0,
    double emergency_vram_pct = 93.0, double min_headroom_pct = 4.0,
    double runtime_overhead_pct = 5.0, int min_headroom_floor_mib = 512,
    int overhead_floor_mib = 256, int target_mib_override = 0, int headroom_mib_override = 0,
    int emergency_mib_override = 0, int min_headroom_mib_override = 0,
    int runtime_overhead_mib_override = 0);

}  // namespace xn
