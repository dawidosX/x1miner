#pragma once

#include "common.hpp"
#include "config/settings.hpp"
#include "efficiency/thermal_policy.hpp"
#include "efficiency/vram_policy.hpp"
#include "hashapi/HashApiTypes.h"
#include "mining/vram_batch.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace xn {

struct MineBatchResult {
    int64_t hashes_done = 0;
    /// All valid matches this wave (multi-lane × multi-match). Prefer this over hit.
    std::vector<BlockHit> hits;
    /// First/best hit for callers that only want one (kept for compatibility).
    std::optional<BlockHit> hit;
    bool aborted = false;
    std::string abort_reason;
};

class CudaEngine {
public:
    explicit CudaEngine(const Settings& settings);
    ~CudaEngine();

    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;

    void start();
    void stop();
    bool is_running() const { return started_; }

    void set_vram_caps(const VramCaps& caps);
    /// Supervisor compatibility: champ planner does not subtract live NVML.
    void set_occupied_vram_mib(int /*mib*/) {}
    /// Supervisor compatibility: keep the current pack (thermal scale handles heat).
    double note_vram_emergency() { return 1.0; }
    double vram_pack_scale() const { return 1.0; }
    void set_difficulty(int difficulty);
    int difficulty() const { return difficulty_; }
    /// Queue pressure for value-priority XUNI throttling (from BlockStore counts).
    void set_queue_counts(int pending_xuni, int pending_xnm, int pending_xblk);
    bool xuni_mining_active() const { return last_allow_xuni_; }

    MineBatchResult mine_batch();
    /// Finalize any double-buffered results still sitting on lanes (call before stop).
    MineBatchResult drain_pipeline();
    // Last aggregate hashrate across lanes (H/s).
    double last_hashrate() const { return last_hashrate_; }
    // Champ harvest has no separate kernel counter; dashboard uses wall H/s.
    double last_kernel_hashrate() const { return last_hashrate_; }

    int batch_size() const { return batch_per_lane_; }
    int active_lanes() const { return lanes_; }
    int max_lanes_cap() const { return max_lanes_cap_; }
    int reduce_lane_cap();
    int add_lane_cap();
    bool restore_lane_cap_if_cool(int temperature_c, int difficulty, int memory_junction_c = 0);
    int thermal_idle_ms() const { return thermal_idle_ms_; }
    double set_thermal_batch_scale(double scale);
    ThermalHuntResult update_thermal_batch_from_temp(int temperature_c, int memory_junction_c = 0,
                                                     double now_s = 0);
    double thermal_batch_scale() const { return thermal_batch_scale_; }
    const CudaVramPlan* vram_plan() const {
        return vram_plan_ ? &*vram_plan_ : nullptr;
    }
    std::string device_name() const { return device_name_; }
    uint64_t total_vram_bytes() const { return total_vram_bytes_; }
    uint64_t free_vram_bytes() const;

private:
    struct Lane;
    struct LaneMailbox {
        std::atomic<int> seq{0};
        std::atomic<int> done_seq{0};
        std::atomic<int> produced{0};
        std::atomic<int> consumed{0};
        int role = 0;
        uint32_t difficulty = 0;
        size_t batch_size = 0;
        size_t arena_batch_size = 0;
        int xuni_lanes = 0;
        bool allow_xuni_base = false;
        int device_id = 0;
        int work_patches = 2;
        std::string salt;
        static constexpr int kRing = 4;
        hashapi::HashApiResult ring[kRing];
        std::string ring_err[kRing];
    };

    static constexpr int kMaxLaneMailboxes = 16;
    static constexpr int kMaxGroups = 8;
    static constexpr int kInnerSwaps = 256;
    static constexpr double kInnerSeconds = 1.5;

    void replan(int difficulty);
    void sync_lanes();
    void ensure_lane_workers();
    void stop_lane_workers();
    void lane_worker_loop(int lane_index);
    void build_groups();
    void dispatch_group(int group, int role);
    void wait_group(int group);
    void absorb_group(MineBatchResult& out, double& total_hs, int group);
    void reap_creates();
    int pick_ready(int exclude) const;
    bool group_lanes_done(int group) const;
    std::optional<BlockHit> hit_from_match(const std::string& key, const std::string& hash,
                                           const std::string& pattern, int64_t attempts,
                                           double hps) const;

    Settings settings_;
    bool started_ = false;
    int difficulty_ = 1100;
    int lanes_ = 1;
    int batch_per_lane_ = 0;
    int planned_batch_per_lane_ = 0;
    int work_patches_ = 2;
    int hash_patch_ = 0;
    int n_groups_ = 0;
    int group_lo_[kMaxGroups] = {};
    int group_hi_[kMaxGroups] = {};
    bool group_ready_[kMaxGroups] = {};
    bool group_creating_[kMaxGroups] = {};
    bool group_launched_[kMaxGroups] = {};
    size_t group_launch_batch_[kMaxGroups] = {};
    int inflight_group_ = -1;
    size_t pipeline_batch_ = 0;
    int max_lanes_cap_ = 8;
    int config_max_lanes_ = 8;
    double thermal_batch_scale_ = 1.0;
    int thermal_idle_ms_ = 0;
    ThermalHuntState thermal_hunt_{};
    std::optional<VramCaps> vram_caps_;
    std::optional<CudaVramPlan> vram_plan_;
    std::string device_name_;
    uint64_t total_vram_bytes_ = 0;
    std::vector<std::unique_ptr<Lane>> lanes_impl_;
    std::mutex mu_;
    double last_hashrate_ = 0.0;

    // Persistent per-lane workers (avoid std::async create/join every batch).
    std::vector<std::thread> lane_workers_;
    std::mutex worker_mu_;
    std::condition_variable worker_cv_;
    std::condition_variable worker_done_cv_;
    std::atomic<bool> workers_stop_{false};
    int worker_target_lanes_ = 0;
    std::array<LaneMailbox, kMaxLaneMailboxes> mailboxes_;
    std::vector<hashapi::HashApiResult> lane_results_;
    std::vector<std::string> lane_errors_;
    std::string proto_salt_;
    uint32_t proto_difficulty_ = 0;
    size_t proto_batch_size_ = 0;
    size_t proto_arena_batch_ = 0;
    int proto_xuni_lanes_ = 0;
    bool proto_allow_xuni_base_ = false;
    int proto_device_id_ = 0;
    int proto_work_patches_ = 2;

    int pending_xuni_ = 0;
    int pending_xnm_ = 0;
    int pending_xblk_ = 0;
    int batch_seq_ = 0;
    bool xuni_queue_paused_ = false;
    bool last_allow_xuni_ = false;

    bool compute_allow_xuni_base();
};

}  // namespace xn
