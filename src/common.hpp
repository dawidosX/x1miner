#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace xn {

inline constexpr const char* kMinerVersion = "5.1.0-m10000-parked";
inline constexpr const char* kAppName = "XNMiner CUDA";
inline constexpr const char* kAppTagline = "x1miner";
// Wrapper (vast.sh) treats this as "bag persisted, pull + rebuild + restart".
inline constexpr int kExitCodeUpdate = 75;

struct BlockHit {
    std::string key;
    std::string hash_str;
    std::string block_type;
    int64_t attempts = 0;
    std::string strategy = "random";
    double hps = 0.0;
    std::string found_at;  // ISO local time
    std::optional<int> memory_cost;
};

struct GpuSnapshot {
    int index = 0;
    std::string name;
    int total_mib = 0;
    int used_mib = 0;
    int free_mib = 0;
    int util_pct = 0;
    double power_w = -1.0;
    int temperature_c = 0;          // GPU die / edge (°C)
    int memory_junction_c = 0;      // GDDR memory junction (°C). 0 = sensor unavailable.
    bool has_memory_junction() const { return memory_junction_c > 0; }
};

struct NetworkStatus {
    bool ok = false;
    std::optional<int> difficulty;
    double latency_ms = -1.0;
    std::string error;
    uint64_t seq = 0;  // poller generation; 0 = no poll yet
};

struct MiningStats {
    int64_t total_hashes = 0;
    int session_hits = 0;
    int active_lanes = 0;
    double hps_ema = 0.0;

    int found_xuni = 0, found_xnm = 0, found_xblk = 0;
    int enqueued_xuni = 0, enqueued_xnm = 0, enqueued_xblk = 0;
    int accepted_live_xuni = 0, accepted_live_xnm = 0, accepted_live_xblk = 0;
    int accepted_flush_xuni = 0, accepted_flush_xnm = 0, accepted_flush_xblk = 0;
    int failed_live_xuni = 0, failed_live_xnm = 0, failed_live_xblk = 0;
    int rejected_flush_xuni = 0, rejected_flush_xnm = 0, rejected_flush_xblk = 0;
    int rejected_live_xuni = 0, rejected_live_xnm = 0, rejected_live_xblk = 0;
    int resubmission_xuni = 0, resubmission_xnm = 0, resubmission_xblk = 0;
    int dropped_flush = 0;
    int queued = 0;

    int found_total() const { return found_xuni + found_xnm + found_xblk; }
    int accepted_live_total() const {
        return accepted_live_xuni + accepted_live_xnm + accepted_live_xblk;
    }
    int accepted_flush_total() const {
        return accepted_flush_xuni + accepted_flush_xnm + accepted_flush_xblk;
    }
    int accepted_total() const { return accepted_live_total() + accepted_flush_total(); }
    int accepted_xnm_total() const { return accepted_live_xnm + accepted_flush_xnm; }
    // Session XNM production rate: found this session / uptime.
    // Uses found (not accepted-only) so queued XNM still counts while waiting to submit.
    double avg_xnm_per_s(int uptime_s) const {
        if (uptime_s <= 0) return 0.0;
        return static_cast<double>(found_xnm) / static_cast<double>(uptime_s);
    }
    int rejected_live_total() const {
        return rejected_live_xuni + rejected_live_xnm + rejected_live_xblk;
    }
    int rejected_flush_total() const {
        return rejected_flush_xuni + rejected_flush_xnm + rejected_flush_xblk;
    }
    int rejected_total() const { return rejected_live_total() + rejected_flush_total(); }
    int resubmission_total() const {
        return resubmission_xuni + resubmission_xnm + resubmission_xblk;
    }
    int enqueued_total() const {
        return enqueued_xuni + enqueued_xnm + enqueued_xblk;
    }
};

struct SubmitResult {
    bool ok = false;
    int status = 0;
    std::string body;
    std::string submitted_at;
};

inline int& stat_field_for_kind(MiningStats& s, const std::string& kind, const char* bucket) {
    // bucket: found|enqueued|accepted_live|accepted_flush|rejected_live|rejected_flush|resubmission
    const bool xuni = kind == "XUNI";
    const bool xblk = kind == "XBLK";
    if (std::string(bucket) == "found") {
        return xuni ? s.found_xuni : (xblk ? s.found_xblk : s.found_xnm);
    }
    if (std::string(bucket) == "enqueued") {
        return xuni ? s.enqueued_xuni : (xblk ? s.enqueued_xblk : s.enqueued_xnm);
    }
    if (std::string(bucket) == "accepted_live") {
        return xuni ? s.accepted_live_xuni : (xblk ? s.accepted_live_xblk : s.accepted_live_xnm);
    }
    if (std::string(bucket) == "accepted_flush") {
        return xuni ? s.accepted_flush_xuni : (xblk ? s.accepted_flush_xblk : s.accepted_flush_xnm);
    }
    if (std::string(bucket) == "rejected_live") {
        return xuni ? s.rejected_live_xuni : (xblk ? s.rejected_live_xblk : s.rejected_live_xnm);
    }
    if (std::string(bucket) == "rejected_flush") {
        return xuni ? s.rejected_flush_xuni : (xblk ? s.rejected_flush_xblk : s.rejected_flush_xnm);
    }
    if (std::string(bucket) == "resubmission") {
        return xuni ? s.resubmission_xuni : (xblk ? s.resubmission_xblk : s.resubmission_xnm);
    }
    return s.found_xnm;
}

}  // namespace xn
