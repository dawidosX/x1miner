#include "util/cpu.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <string>
#include <thread>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace xn {
namespace {

std::string trim_ws(std::string s) {
    auto not_space = [](unsigned char c) { return c != ' ' && c != '\t' && c != '\r'; };
    while (!s.empty() && !not_space(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && !not_space(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

}  // namespace

int cpu_logical_count() {
    unsigned n = std::thread::hardware_concurrency();
    if (n == 0) {
#ifndef _WIN32
        const long s = sysconf(_SC_NPROCESSORS_ONLN);
        if (s > 0) n = static_cast<unsigned>(s);
#endif
    }
    if (n == 0) n = 2;
    return static_cast<int>(n);
}

int cpu_physical_count() {
    const int logical = cpu_logical_count();
#ifdef __linux__
    std::ifstream in("/proc/cpuinfo");
    if (in) {
        std::set<std::string> cores;
        std::string line, phys, core;
        while (std::getline(in, line)) {
            if (line.compare(0, 11, "physical id") == 0) {
                auto p = line.find(':');
                if (p != std::string::npos) phys = trim_ws(line.substr(p + 1));
            } else if (line.compare(0, 7, "core id") == 0) {
                auto p = line.find(':');
                if (p != std::string::npos) core = trim_ws(line.substr(p + 1));
                if (!phys.empty() && !core.empty()) {
                    cores.insert(phys + ":" + core);
                    core.clear();
                }
            }
        }
        if (!cores.empty()) return static_cast<int>(cores.size());
    }
#endif
    // Guess SMT: two threads per core when we cannot read topology.
    return std::max(1, logical / 2);
}

int auto_keygen_threads(int logical, int physical, int max_lanes) {
    logical = std::max(1, logical);
    physical = std::max(1, physical);
    const int leave = (logical >= 6) ? 2 : (logical >= 3 ? 1 : 0);
    const int cap = std::max(2, logical - leave);
    const int ceiling = std::min(8, std::max(4, max_lanes));
    const int want = std::min({physical, cap, ceiling});
    return std::max(2, want);
}

int auto_match_drain_parallel(int logical) {
    logical = std::max(1, logical);
    return std::max(8, std::min(64, logical * 4));
}

}  // namespace xn
