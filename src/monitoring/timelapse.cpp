#include "monitoring/timelapse.hpp"

#include "util/paths.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <fstream>

namespace xn {

SessionTimelapse::SessionTimelapse(std::filesystem::path path, int sample_s)
    : path_(std::move(path)), sample_s_(sample_s) {
    ensure_parent_dir(path_);
}

void SessionTimelapse::maybe_sample(const MiningStats& stats, const GpuSnapshot* gpu, int pending,
                                    bool network_ok) {
    auto now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
                   .count();
    if (now - last_sample_ < sample_s_) return;
    last_sample_ = now;
    nlohmann::json j = {{"ts", now_iso_local()},
                        {"hps", stats.hps_ema},
                        {"hashes", stats.total_hashes},
                        {"accepted", stats.accepted_total()},
                        {"rejected", stats.rejected_total()},
                        {"queued", pending},
                        {"network_ok", network_ok}};
    if (gpu) {
        j["temp_c"] = gpu->temperature_c;
        j["memory_junction_c"] = gpu->memory_junction_c;
        j["vram_used_mib"] = gpu->used_mib;
        j["power_w"] = gpu->power_w;
    }
    std::ofstream out(path_, std::ios::app);
    out << j.dump() << "\n";
}

void SessionTimelapse::record_event(const std::string& label) {
    nlohmann::json j = {{"ts", now_iso_local()}, {"event", label}};
    std::ofstream out(path_, std::ios::app);
    out << j.dump() << "\n";
}

void SessionTimelapse::finalize() {
    nlohmann::json j = {{"ts", now_iso_local()}, {"event", "session_end"}};
    std::ofstream out(path_, std::ios::app);
    out << j.dump() << "\n";
}

}  // namespace xn
