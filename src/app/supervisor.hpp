#pragma once

#include "common.hpp"
#include "config/settings.hpp"
#include "efficiency/gpu_power.hpp"
#include "mining/cuda_engine.hpp"
#include "monitoring/dashboard.hpp"
#include "monitoring/local_stats.hpp"
#include "monitoring/metrics.hpp"
#include "monitoring/nvml_monitor.hpp"
#include "monitoring/timelapse.hpp"
#include "monitoring/wallet.hpp"
#include "monitoring/woodyminer.hpp"
#include "monitoring/xenblockscan.hpp"
#include "network/bag_forward.hpp"
#include "network/difficulty.hpp"
#include "network/lastblock.hpp"
#include "network/submitter.hpp"
#include "network/updater.hpp"
#include "queue/store.hpp"
#include "app/instance_lock.hpp"
#include "util/logger.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace xn {

class Supervisor {
public:
    Supervisor(Settings settings, bool use_dashboard);
    ~Supervisor();

    bool startup_checks();
    void run(std::optional<int> max_seconds = std::nullopt);
    void request_stop();
    /// Persist all in-flight / pending blocks to disk for next boot (safe from console handler).
    void persist_queue_for_restart();
    /// True after run()/finalize completed clean shutdown.
    bool shutdown_complete() const { return finalized_; }
    /// Wrapper should git pull + rebuild + restart (queue already bagged).
    bool update_requested() const { return update_requested_.load(); }

private:
    struct LiveSubmitJob {
        BlockHit hit;
        std::string kind;
    };

    void log(const std::string& level, const std::string& msg);
    // replan_engine: only the mining thread may replan CUDA difficulty/VRAM.
    bool refresh_network(bool blocking = false, bool replan_engine = false);
    void refresh_paper();
    bool oracle_says_m(int m) const;
    bool oracle_left_m(int m) const;
    int submit_target_m() const;
    int mining_difficulty() const;
    void handle_hit(BlockHit hit);
    void queue_hit(const BlockHit& hit, const std::string& kind, const std::string& reason,
                   const std::string& retry_when);
    void enqueue_live_submit(BlockHit hit, std::string kind);
    void process_live_submit(BlockHit hit, std::string kind);
    int try_flush_pending(const std::string& context, bool on_shutdown = false);
    void service_pending_queue(double now);
    void start_submit_worker();
    void stop_submit_worker();
    void submit_worker_loop();
    /// Move live CPU-submit queue + pending store to disk (no HTTP).
    int bag_live_queue_to_store(const std::string& reason);
    void graceful_shutdown(const std::string& reason);
    void finalize_session();
    void ui_refresh();
    void ui_event(const std::string& action, const std::string& block, const std::string& detail = "");
    void apply_vram_policy();
    bool gpu_safety_tick(const GpuSnapshot* snap);

    /// Target m= we bag for (force-mine 100, or live network m= in classic mode).
    int bag_target_m() const;
    /// How many disk-queue hits can submit at current network m=.
    int matching_queue_depth() const;
    /// Enter/exit aggressive HTTP flush when paper or thermometer shows bag m=.
    void update_match_drain(double now);
    bool match_drain_active() const;
    /// Park CUDA only when live /difficulty is bag m=. Mixed paper windows keep mining.
    bool should_park_cuda() const;
    void handle_batch_hits(MineBatchResult& batch);

    Settings settings_;
    bool use_dashboard_ = true;
    std::unique_ptr<SessionLogger> logger_;
    std::unique_ptr<InstanceLock> lock_;
    std::unique_ptr<MetricsTracker> metrics_;
    std::unique_ptr<NvmlMonitor> gpu_;
    std::unique_ptr<BlockStore> store_;
    std::unique_ptr<Submitter> submitter_;
    std::unique_ptr<BagForwarder> bag_forward_;
    std::unique_ptr<CudaEngine> engine_;
    std::unique_ptr<NetworkPoller> poller_;
    std::unique_ptr<LastblockPoller> paper_;
    std::unique_ptr<MinerDashboard> dashboard_;
    std::unique_ptr<GpuPowerBooster> power_;
    std::unique_ptr<WoodyminerUploader> woody_;
    std::unique_ptr<WalletBalanceTracker> wallet_;
    std::unique_ptr<SessionTimelapse> timelapse_;
    std::unique_ptr<LocalMiningStatsTracker> local_stats_;
    XenblockscanReporter xbs_;

    std::atomic<bool> running_{false};
    std::atomic<bool> shutting_down_{false};
    std::atomic<bool> update_requested_{false};
    double last_update_check_ = 0;
    std::mutex persist_mu_;  // serialize bag/persist from console handler + main

    // Network / submit state (shared: mining thread + CPU submit worker)
    mutable std::mutex state_mu_;
    bool network_ok_ = false;
    bool difficulty_stale_ = false;  // last-good diff while /difficulty flaky
    std::optional<int> network_difficulty_;
    double last_difficulty_ok_at_ = 0;
    int difficulty_fail_streak_ = 0;
    uint64_t last_poll_seq_ = 0;
    bool paper_ok_ = false;
    std::optional<int> paper_newest_m_;
    int paper_tip_id_ = 0;
    double last_paper_ok_at_ = 0;
    uint64_t last_paper_seq_ = 0;
    double defer_submit_until_ = 0;
    double submit_backoff_until_ = 0;  // after transport fail: queue only, keep mining

    double cooldown_until_ = 0;
    double last_queue_flush_ = 0;
    double last_xuni_flush_ = 0;
    double last_flush_skip_log_at_ = 0;
    double last_ui_ = 0;
    double last_xbs_holdings_ = 0;
    double session_started_at_ = 0;
    bool was_in_xuni_window_ = false;
    bool reduce_lanes_after_cooldown_ = false;
    bool finalized_ = false;
    std::optional<VramCaps> vram_caps_;

    // Match-drain: CPU flushes while paper/thermometer shows bag m=.
    // CUDA parks only on live m= match, not to slow bag growth.
    std::atomic<bool> match_drain_active_{false};
    bool match_drain_gpu_parked_ = false;
    double match_drain_until_ = 0;
    int match_drain_start_queue_ = 0;
    std::atomic<int> match_drain_flushed_{0};
    bool match_drain_logged_first_wave_ = false;
    int64_t flush_skip_before_id_ = 0;

    // CPU submit worker — all HTTP /verify and queue flush happens here.
    std::thread submit_thread_;
    std::mutex submit_mu_;
    std::condition_variable submit_cv_;
    std::deque<LiveSubmitJob> live_submit_q_;
    std::atomic<bool> submit_worker_running_{false};

    bool live_submit_allowed() const;
    /// Queue flush gate: paper or last-good m= is enough (flaky /difficulty must not freeze the bag).
    bool flush_submit_allowed() const;
    bool network_matches_hit_m(int hit_m) const;
    bool can_submit_hit_m(int hit_m) const;
    bool force_mine_mode() const { return settings_.force_mine_memory_cost > 0; }
    int forced_mine_m() const { return settings_.force_mine_memory_cost; }
    /// Live network m= for public stats (Woody). 0 = unknown / N/A — never mine m=.
    int network_difficulty_for_public() const;
    void note_submit_transport_failure(const char* where, int status = 0);
    void log_flush_skip(const std::string& why);
    int live_submit_timeout_s() const;
    void maybe_check_update(double now);
};

}  // namespace xn
