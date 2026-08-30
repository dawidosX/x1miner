#pragma once

namespace xn {

/// Online logical CPUs (hardware threads), at least 1.
int cpu_logical_count();

/// Distinct physical cores from /proc/cpuinfo. Falls back if unavailable.
int cpu_physical_count();

/// Keygen worker count that will not oversubscribe the machine.
/// Leaves 1–2 logical CPUs for mining, submit, and the OS. Caps at 8.
int auto_keygen_threads(int logical, int physical, int max_lanes);

/// Parallel /verify workers for an m=100 flush. Caps with core count.
int auto_match_drain_parallel(int logical);

}  // namespace xn
