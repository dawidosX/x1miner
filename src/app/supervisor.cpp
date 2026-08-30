#include "app/supervisor.hpp"

#include "efficiency/thermal_policy.hpp"
#include "efficiency/vram_policy.hpp"
#include "mining/argon2_encode.hpp"
#include "mining/block_types.hpp"
#include "queue/policy.hpp"
#include "util/paths.hpp"

#ifdef _WIN32
#ifdef _WIN32
#include <windows.h>
#endif
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

namespace xn {
namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace

Supervisor::Supervisor(Settings settings, bool use_dashboard)
    : settings_(std::move(settings)), use_dashboard_(use_dashboard) {
    logger_ = std::make_unique<SessionLogger>(settings_.log_path, !use_dashboard_);
    lock_ = std::make_unique<InstanceLock>(settings_.log_path.parent_path() / "miner.lock");
    metrics_ = std::make_unique<MetricsTracker>();
    gpu_ = std::make_unique<NvmlMonitor>(settings_.device_id);
    store_ = std::make_unique<BlockStore>(settings_.db_path, settings_.jsonl_path,
                                          settings_.rejected_jsonl_path);
    submitter_ = std::make_unique<Submitter>(settings_.verify_url(), settings_.address,
                                             settings_.worker, logger_.get());
    engine_ = std::make_unique<CudaEngine>(settings_);
    poller_ = std::make_unique<NetworkPoller>(
        settings_.difficulty_url(), settings_.network_poll_interval_s,
        settings_.network_down_poll_interval_s, settings_.network_poll_timeout_s);
    paper_ = std::make_unique<LastblockPoller>(
        settings_.lastblock_url, settings_.lastblock_url_fallback,
        settings_.lastblock_poll_interval_s, settings_.lastblock_timeout_s);
    local_stats_ = std::make_unique<LocalMiningStatsTracker>(settings_.log_path.parent_path() /
                                                             "mining_stats_history.json");
    timelapse_ =
        std::make_unique<SessionTimelapse>(settings_.timelapse_path, settings_.timelapse_sample_s);

    if (use_dashboard_) {
        dashboard_ = std::make_unique<MinerDashboard>(settings_);
    }
    wallet_ = std::make_unique<WalletBalanceTracker>(
        settings_.address, settings_.log_path.parent_path() / "balance_history.json");

    if (settings_.gpu_power_boost_enabled) {
        power_ = std::make_unique<GpuPowerBooster>(
            *gpu_, settings_.gpu_power_target_pct, settings_.warn_gpu_temp_c, settings_.max_gpu_temp_c,
            logger_.get(), settings_.gpu_windows_performance_mode, settings_.gpu_power_min_pct,
            settings_.gpu_difficulty_power_enabled, settings_.vram_reference_difficulty,
            settings_.gpu_difficulty_power_full_ratio, settings_.warn_mem_temp_c,
            settings_.max_mem_temp_c, settings_.thermal_use_memory_junction);
    }

    xbs_.configure(settings_.xenblockscan_enabled && !settings_.address.empty(),
                   settings_.xenblockscan_endpoint, settings_.xenblockscan_api_key,
                   settings_.xenblockscan_report_rejects);
}

Supervisor::~Supervisor() {
    request_stop();
    finalize_session();
}

void Supervisor::log(const std::string& level, const std::string& msg) {
    if (level == "warn") logger_->warn(msg);
    else if (level == "error") logger_->error(msg);
    else logger_->info(msg);
}

void Supervisor::request_stop() { running_ = false; }

int Supervisor::bag_live_queue_to_store(const std::string& reason) {
    std::deque<LiveSubmitJob> batch;
    {
        std::lock_guard<std::mutex> lock(submit_mu_);
        batch.swap(live_submit_q_);
    }
    int n = 0;
    for (auto& job : batch) {
        store_->enqueue(job.hit, reason);
        metrics_->record_enqueued(job.kind);
        ++n;
    }
    if (n) {
        metrics_->sync_pending(store_->pending_count());
        if (bag_forward_) bag_forward_->notify();
    }
    return n;
}

void Supervisor::persist_queue_for_restart() {
    // Safe to call from console Ctrl handler: no long HTTP, just disk bag.
    std::lock_guard<std::mutex> guard(persist_mu_);
    shutting_down_ = true;
    running_ = false;
    submit_worker_running_ = false;
    submit_cv_.notify_all();

    int bagged = bag_live_queue_to_store(SHUTDOWN_PENDING_REASON);
    if (store_) {
        store_->defer_all_to_next_start();
        log("info", "Bagged queue for next start (" + std::to_string(bagged) +
                        " live + " + std::to_string(store_->pending_count()) +
                        " pending on disk)");
    }
}

bool Supervisor::startup_checks() {
    if (!lock_->acquire()) {
        log("error", "Another miner instance is already running (miner.lock).");
        std::cerr << "ERROR: Another miner instance is already running.\n";
        return false;
    }
    if (settings_.address.empty()) {
        log("error", "No wallet address configured");
        return false;
    }
    return true;
}

void Supervisor::apply_vram_policy() {
    auto snap = gpu_->snapshot();
    int total = 0;
    if (snap) total = snap->total_mib;
    if (total <= 0 && engine_) {
        total = static_cast<int>(engine_->total_vram_bytes() / (1024 * 1024));
    }
    if (total <= 0) return;
    vram_caps_ = resolve_vram_caps(
        total, settings_.target_vram_pct, settings_.desktop_headroom_pct,
        settings_.emergency_vram_pct, settings_.min_headroom_pct, settings_.runtime_overhead_pct,
        settings_.min_headroom_floor_mib, settings_.runtime_overhead_floor_mib,
        settings_.target_vram_mib, settings_.headroom_mib, settings_.emergency_vram_mib,
        settings_.min_headroom_mib, settings_.cuda_runtime_overhead_mib);
    engine_->set_vram_caps(*vram_caps_);
    // Only sample occupied VRAM while the pack is down — a live batch is not "baseline".
    if (!engine_->is_running() && snap && snap->used_mib > 0) {
        engine_->set_occupied_vram_mib(snap->used_mib);
    }
    log("info", vram_caps_->summary());
}

int Supervisor::mining_difficulty() const {
    // Hybrid: CUDA always at force_mine_memory_cost (e.g. 100). Network m= is separate.
    if (force_mine_mode()) return forced_mine_m();
    if (engine_) return engine_->difficulty();
    std::lock_guard<std::mutex> lock(state_mu_);
    return network_difficulty_.value_or(settings_.memory_cost);
}

bool Supervisor::live_submit_allowed() const {
    // Mining always continues. Live HTTP submit is optional when pool is unreachable.
    std::lock_guard<std::mutex> lock(state_mu_);
    if (!network_ok_) return false;
    const double t = now_s();
    if (t < submit_backoff_until_) return false;
    // In hybrid mode we do NOT defer submits just because network m= moved — mining m= is fixed.
    if (!force_mine_mode() && t < defer_submit_until_) return false;
    return true;
}

bool Supervisor::flush_submit_allowed() const {
    // Queue flush may proceed on last-good m= even when /difficulty is flaky.
    // Lastblock (paper) is enough to flush — do not require port 80.
    std::lock_guard<std::mutex> lock(state_mu_);
    const double t = now_s();
    if (t < submit_backoff_until_) return false;
    if (!force_mine_mode() && t < defer_submit_until_) return false;
    if (paper_ok_ || network_ok_) return true;
    if (force_mine_mode() && network_difficulty_ && *network_difficulty_ == forced_mine_m()) {
        return true;
    }
    return false;
}

bool Supervisor::oracle_says_m(int m) const {
    if (m <= 0) return false;
    std::lock_guard<std::mutex> lock(state_mu_);
    if (network_ok_ && network_difficulty_ && *network_difficulty_ == m) return true;
    if (paper_ok_ && paper_newest_m_ && *paper_newest_m_ == m) return true;
    return false;
}

bool Supervisor::oracle_left_m(int m) const {
    if (m <= 0) return false;
    std::lock_guard<std::mutex> lock(state_mu_);
    const bool net_says = network_ok_ && network_difficulty_ && *network_difficulty_ == m;
    const bool paper_says = paper_ok_ && paper_newest_m_ && *paper_newest_m_ == m;
    if (net_says || paper_says) return false;
    const bool have_oracle =
        (network_ok_ && network_difficulty_ && *network_difficulty_ > 0) ||
        (paper_ok_ && paper_newest_m_ && *paper_newest_m_ > 0);
    return have_oracle;
}

int Supervisor::submit_target_m() const {
    const int bag = bag_target_m();
    std::lock_guard<std::mutex> lock(state_mu_);
    // Prefer live /difficulty as primary; paper (lastblock) is fallback only. (KIMI)
    if (network_ok_ && network_difficulty_ && *network_difficulty_ > 0) {
        return (*network_difficulty_ == bag) ? bag : *network_difficulty_;
    }
    if (paper_ok_ && paper_newest_m_ && *paper_newest_m_ > 0) {
        return (*paper_newest_m_ == bag) ? bag : *paper_newest_m_;
    }
    return network_difficulty_.value_or(0);
}

bool Supervisor::network_matches_hit_m(int hit_m) const { return oracle_says_m(hit_m); }

bool Supervisor::can_submit_hit_m(int hit_m) const {
    return live_submit_allowed() && network_matches_hit_m(hit_m);
}

int Supervisor::bag_target_m() const {
    // Hybrid: we always bag at force-mine m= (100). Classic: bag at whatever we mine.
    if (force_mine_mode()) return forced_mine_m();
    std::lock_guard<std::mutex> lock(state_mu_);
    return network_difficulty_.value_or(settings_.memory_cost);
}

int Supervisor::matching_queue_depth() const {
    int net_m = submit_target_m();
    if (net_m <= 0 || !store_) return 0;
    return store_->pending_eligible_m(net_m, bag_target_m());
}

bool Supervisor::match_drain_active() const {
    return match_drain_active_.load();
}

bool Supervisor::should_park_cuda() const {
    if (!match_drain_active_.load()) return false;
    const int bag = bag_target_m();
    std::lock_guard<std::mutex> lock(state_mu_);
    if (network_ok_ && network_difficulty_ && *network_difficulty_ > 0) {
        return *network_difficulty_ == bag;
    }
    return paper_ok_ && paper_newest_m_ && *paper_newest_m_ == bag;
}

void Supervisor::refresh_paper() {
    if (!paper_) return;
    PaperStatus st = paper_->get_status();
    if (st.seq == 0 || st.seq == last_paper_seq_) return;
    last_paper_seq_ = st.seq;
    int old_m = 0;
    bool changed = false;
    {
        std::lock_guard<std::mutex> lock(state_mu_);
        old_m = paper_newest_m_.value_or(0);
        if (st.ok && st.newest_m) {
            paper_ok_ = true;
            paper_newest_m_ = st.newest_m;
            paper_tip_id_ = st.tip_id;
            last_paper_ok_at_ = now_s();
            changed = old_m > 0 && old_m != *st.newest_m;
        } else {
            const double age = last_paper_ok_at_ > 0 ? (now_s() - last_paper_ok_at_) : 1e9;
            paper_ok_ = paper_newest_m_.has_value() && age < 30.0;
        }
    }
    if (changed) {
        log("info", "Paper newest m=" + std::to_string(old_m) + " -> " +
                        std::to_string(*st.newest_m) + " (tip " + std::to_string(st.tip_id) +
                        (st.mixed ? ", mixed window)" : ")"));
        submit_cv_.notify_all();
    }
    if (dashboard_ && st.ok && st.newest_m) {
        bool diff_ok = false;
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            diff_ok = network_ok_;
        }
        if (!diff_ok) dashboard_->set_network(true, st.newest_m, true);
        if (force_mine_mode()) dashboard_->set_mining_m(forced_mine_m(), true);
    }
}

void Supervisor::update_match_drain(double now) {
    // CPU flush while paper or thermometer shows bag m=. CUDA parks only on live match.
    if (!settings_.match_drain_enabled) {
        if (match_drain_active_) {
            match_drain_active_ = false;
            match_drain_gpu_parked_ = false;
            log("info", "Match-flush off (disabled in config)");
            if (dashboard_) dashboard_->set_status("Mining");
        }
        return;
    }

    int net_m = 0;
    int paper_m = 0;
    int paper_tip = 0;
    {
        std::lock_guard<std::mutex> lock(state_mu_);
        net_m = network_difficulty_.value_or(0);
        paper_m = paper_newest_m_.value_or(0);
        paper_tip = paper_tip_id_;
    }
    const int bag_m = bag_target_m();
    const bool last_good_matches = oracle_says_m(bag_m);
    const bool m_match = last_good_matches;
    const bool confirmed_mismatch = oracle_left_m(bag_m);
    // Eligible = hit m >= bag m: parked higher-m finds flush too when diff is at/below their m.
    const int bag_q = store_ ? store_->pending_eligible_m(bag_m, bag_m) : 0;
    const int match_q = last_good_matches ? bag_q : 0;
    const int min_q = settings_.match_drain_min_queue;
    const int shown_m = (net_m == bag_m) ? net_m : (paper_m > 0 ? paper_m : net_m);

    if (match_drain_active_) {
        const bool timed_out =
            settings_.match_drain_max_s > 0 && match_drain_until_ > 0 && now >= match_drain_until_;
        const bool empty = last_good_matches && bag_q <= 0;
        if (empty || timed_out || confirmed_mismatch) {
            const char* why =
                confirmed_mismatch ? "oracles left bag m=" : (empty ? "queue clear" : "time budget");
            log("info", std::string("Match-flush END (") + why + ") — flushed~" +
                            std::to_string(match_drain_flushed_.load()) +
                            " this window, remaining bag_q=" + std::to_string(bag_q) +
                            (last_good_matches ? "" : " (net not matching)") +
                            " — CUDA resumes mining m=" + std::to_string(bag_m));
            match_drain_active_ = false;
            match_drain_gpu_parked_ = false;
            match_drain_until_ = 0;
            if (dashboard_) {
                dashboard_->set_status(std::string("Mining m=") + std::to_string(bag_m));
            }
            if (store_) store_->flush();
            submit_cv_.notify_all();
        } else {
            submit_cv_.notify_all();
            if (dashboard_) {
                dashboard_->set_status(std::string("FLUSH m=") + std::to_string(shown_m) +
                                       " · " + std::to_string(match_q) + " blocks · " +
                                       (should_park_cuda() ? "GPU parked" : "GPU mining"));
            }
        }
        return;
    }

    if (m_match && match_q >= min_q) {
        match_drain_active_ = true;
        match_drain_gpu_parked_ = false;
        match_drain_until_ = settings_.match_drain_max_s > 0
                                 ? now + settings_.match_drain_max_s
                                 : 0;
        match_drain_start_queue_ = match_q;
        match_drain_flushed_ = 0;
        match_drain_logged_first_wave_ = false;
        flush_skip_before_id_ = 0;
        const bool mixed = (net_m > 0 && paper_m > 0 && net_m != paper_m);
        const bool park = should_park_cuda();
        log("info", "Match-flush START — bag m=" + std::to_string(bag_m) +
                        " (diff m=" + std::to_string(net_m) + " paper newest m=" +
                        std::to_string(paper_m) + " tip=" + std::to_string(paper_tip) +
                        (mixed ? ", mixed — stay until paper leaves" : "") +
                        ") q=" + std::to_string(match_q) +
                        (park ? " — parking CUDA, CPU drains /verify"
                              : " — CUDA keeps mining, CPU drains /verify"));
        if (dashboard_) {
            dashboard_->set_status(std::string("FLUSH m=") + std::to_string(shown_m) + " · " +
                                   std::to_string(match_q) + " blocks · " +
                                   (park ? "GPU parked" : "GPU mining"));
        }
        submit_cv_.notify_all();
    }
}

int Supervisor::network_difficulty_for_public() const {
    // Private: never report hybrid force-mine m= to leaderboards.
    // Prefer /difficulty; if port 80 is down, sealed lastblock newest m= is the truth.
    std::lock_guard<std::mutex> lock(state_mu_);
    if (network_ok_ && network_difficulty_) return *network_difficulty_;
    if (paper_ok_ && paper_newest_m_) return *paper_newest_m_;
    return 0;  // N/A
}

int Supervisor::live_submit_timeout_s() const {
    // CPU submit path: allow full configured timeout (does not block CUDA).
    return settings_.connection_timeout_s > 0 ? settings_.connection_timeout_s : 8;
}

void Supervisor::note_submit_transport_failure(const char* where, int status) {
    // 401/429 during an m=100 match window is pool rate-limit, not "network down".
    // Flipping network_ok_ off here used to abort match-flush (logged as "queue clear")
    // while thousands of m=100 hashes were still sitting in the bag.
    const bool hybrid_match = force_mine_mode() && network_matches_hit_m(forced_mine_m());
    const bool rate_limit = status == 401 || status == 403 || status == 429;
    const double backoff = (hybrid_match || rate_limit)
                               ? 3.0
                               : std::max(15.0, static_cast<double>(settings_.network_down_poll_interval_s));
    std::optional<int> diff;
    bool stale = false;
    bool keep_net = false;
    {
        std::lock_guard<std::mutex> lock(state_mu_);
        submit_backoff_until_ = now_s() + backoff;
        keep_net = hybrid_match || rate_limit;
        if (!keep_net) network_ok_ = false;
        diff = network_difficulty_;
        stale = difficulty_stale_;
    }
    if (dashboard_ && !keep_net) dashboard_->set_network(false, diff, stale);
    log("warn", std::string(where) +
                    (status ? (" HTTP " + std::to_string(status)) : std::string()) +
                    (keep_net ? " — brief submit pause " : " unreachable — mining continues, queueing hits for ") +
                    std::to_string(static_cast<int>(backoff)) + "s");
}

void Supervisor::log_flush_skip(const std::string& why) {
    const double t = now_s();
    if (t - last_flush_skip_log_at_ < 10.0) return;
    last_flush_skip_log_at_ = t;
    log("warn", "Queue flush skipped — " + why);
}

void Supervisor::maybe_check_update(double now) {
    if (!settings_.update_check_enabled) return;
    if (shutting_down_ || update_requested_) return;
    if (match_drain_active()) return;
    if (last_update_check_ <= 0) {
        last_update_check_ = now;  // first check after one full interval
        return;
    }
    if (now - last_update_check_ < static_cast<double>(settings_.update_check_interval_s)) return;
    last_update_check_ = now;

    std::string token = settings_.update_token;
    if (token.empty()) token = github_token_from_env();
    const std::string local = read_build_sha(settings_.update_sha_path.string());
    if (local.empty()) return;
    auto st = check_github_update(settings_.update_github_repo, settings_.update_github_ref, token,
                                  local);
    if (!st.error.empty() && !st.available) {
        log("warn", "Update check failed — " + st.error);
        return;
    }
    if (!st.available) return;
    log("info", "GitHub update " + st.local_sha.substr(0, 8) + " -> " +
                    st.remote_sha.substr(0, 8) + " — bagging queue and restarting into new build");
    update_requested_ = true;
    persist_queue_for_restart();
}

bool Supervisor::refresh_network(bool blocking, bool replan_engine) {
    // API findings: /difficulty is flaky and often needs 8–12s; timeouts must not
    // zero the hashrate path or force-queue every hit. Keep last-good difficulty
    // (never use leaderboard stub "100") and only mark network hard-down after
    // a streak of failures.
    //
    // replan_engine=true only from the mining thread (CUDA set_difficulty is not
    // safe to call from the CPU submit worker).
    NetworkStatus st =
        blocking ? poller_->poll_once(settings_.connection_timeout_s) : poller_->get_status();
    // Non-blocking reads reuse the same poller snapshot. Counting that snapshot
    // every 50ms while CUDA is parked used to burn the fail budget in ~300ms
    // and abort match-flush as "net m= moved".
    {
        std::lock_guard<std::mutex> lock(state_mu_);
        if (!blocking && (st.seq == 0 || st.seq == last_poll_seq_)) {
            return network_ok_ && network_difficulty_.has_value();
        }
        if (st.seq != 0) last_poll_seq_ = st.seq;
    }
    if (st.difficulty) {
        int raw = *st.difficulty;
        int old_diff = 0;
        bool changed = false;
        int diff = 0;
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            int fallback = network_difficulty_.value_or(settings_.memory_cost);
            diff = accept_network_difficulty(raw, fallback);
            if (network_difficulty_ && diff != *network_difficulty_) {
                old_diff = *network_difficulty_;
                changed = true;
                defer_submit_until_ =
                    now_s() + std::max(5.0, static_cast<double>(settings_.sample_interval_s));
            }
            network_difficulty_ = diff;
            network_ok_ = true;
            difficulty_stale_ = false;
            difficulty_fail_streak_ = 0;
            last_difficulty_ok_at_ = now_s();
            // Difficulty is back — allow a live submit probe (backoff cleared).
            submit_backoff_until_ = 0;
        }
        if (changed) {
            if (force_mine_mode()) {
                // Hybrid: CUDA stays on fixed m=; only submit eligibility changes.
                log("info", "Network difficulty " + std::to_string(old_diff) + " -> " +
                                std::to_string(diff) + " (hybrid mine m=" +
                                std::to_string(forced_mine_m()) +
                                (diff == forced_mine_m()
                                     ? " — MATCH, flushing queue when possible)"
                                     : " — queue hits until net m= matches)"));
                if (dashboard_) {
                    dashboard_->set_status(diff == forced_mine_m()
                                               ? "Hybrid: net matches mine m — submitting"
                                               : "Hybrid: mining fixed m, queueing until net matches");
                }
                // Power profile can follow network load; hashing m= does not.
                if (power_) power_->set_difficulty(forced_mine_m());
                submit_cv_.notify_one();
            } else {
                log("info", "Difficulty " + std::to_string(old_diff) + " -> " + std::to_string(diff) +
                                " — queuing hits briefly");
                if (dashboard_) dashboard_->set_status("Lane replan — queuing hits...");
                if (power_) power_->set_difficulty(diff);
                if (replan_engine && engine_ && engine_->is_running()) {
                    try {
                        engine_->set_difficulty(diff);
                        if (auto* plan = engine_->vram_plan()) {
                            log("info", "CUDA VRAM plan: " + plan->summary());
                        }
                    } catch (const std::exception& ex) {
                        log("warn", std::string("Difficulty replan failed: ") + ex.what());
                    }
                }
            }
        } else if (power_) {
            power_->set_difficulty(force_mine_mode() ? forced_mine_m() : diff);
        }
        if (dashboard_) {
            dashboard_->set_network(true, diff, false);
            if (force_mine_mode()) dashboard_->set_mining_m(forced_mine_m(), true);
        }
        return true;
    }

    // Poll failed (timeout / empty). Keep last-good for mining.
    // Submits now live on a dedicated CPU worker, so one flaky /difficulty
    // must not abandon a short hybrid m=100 match window (historically 2–6 min).
    std::optional<int> last_diff;
    bool soft_stale = false;
    bool have_last_good = false;
    {
        std::lock_guard<std::mutex> lock(state_mu_);
        ++difficulty_fail_streak_;
        have_last_good = network_difficulty_.has_value();
        last_diff = network_difficulty_;
        const double age = last_difficulty_ok_at_ > 0 ? (now_s() - last_difficulty_ok_at_) : 1e9;
        const bool matching_hybrid =
            force_mine_mode() && have_last_good && *last_diff == forced_mine_m();
        const int fail_budget = matching_hybrid ? 6 : 3;
        const double age_budget = matching_hybrid ? 120.0 : 60.0;
        if (have_last_good && age < 180.0 && difficulty_fail_streak_ < 8) {
            difficulty_stale_ = true;
            // Keep submit path up through a few timeouts while last-good still
            // matches the bag. Older code flipped network_ok_ off on the first
            // fail and missed entire m=100 flush windows.
            network_ok_ = difficulty_fail_streak_ < fail_budget && age < age_budget;
            soft_stale = true;
        } else {
            network_ok_ = false;
            difficulty_stale_ = have_last_good;
        }
    }
    int fail_streak = 0;
    {
        std::lock_guard<std::mutex> lock(state_mu_);
        fail_streak = difficulty_fail_streak_;
    }
    if (soft_stale) {
        bool still_ok = false;
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            still_ok = network_ok_;
        }
        if (dashboard_) dashboard_->set_network(still_ok, last_diff, true);
        if (fail_streak == 1 || fail_streak % 4 == 0) {
            log("warn", "Difficulty poll failed (" +
                            (st.error.empty() ? "timeout" : st.error) +
                            ") — mining at last-good m=" +
                            std::to_string(last_diff.value_or(settings_.memory_cost)) +
                            (still_ok ? ", last-good still valid for submit"
                                      : ", queueing submits"));
        }
        // Last-good is still inside the fail budget — do not abort flush.
        return still_ok;
    }

    if (dashboard_) dashboard_->set_network(false, last_diff, have_last_good);
    return false;
}

void Supervisor::ui_event(const std::string& action, const std::string& block,
                          const std::string& detail) {
    if (timelapse_) {
        std::string label = action + " " + block;
        if (!detail.empty()) label += " " + detail;
        timelapse_->record_event(label);
    }
    if (dashboard_) {
        dashboard_->event(action, block, detail);
        ui_refresh();
    }
}

void Supervisor::ui_refresh() {
    auto snap = gpu_->snapshot();
    if (wallet_) {
        wallet_->maybe_refresh(false);
        if (dashboard_) dashboard_->set_wallet_line(wallet_->summary_line());
    }
    if (dashboard_ && engine_) {
        dashboard_->set_cuda_batch(engine_->batch_size(), engine_->active_lanes(),
                                   engine_->thermal_batch_scale());
        if (session_started_at_ > 0) {
            dashboard_->set_uptime_s(static_cast<int>(std::max(0.0, now_s() - session_started_at_)));
        }
        dashboard_->update(metrics_->stats(), snap ? &*snap : nullptr,
                           store_->pending_by_type(false), store_->pending_by_type(true));
        dashboard_->render();
    }
    if (timelapse_) {
        bool net_ok = false;
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            net_ok = network_ok_;
        }
        timelapse_->maybe_sample(metrics_->stats(), snap ? &*snap : nullptr, store_->pending_count(),
                                 net_ok);
    }
    // XenBlockScan holdings heartbeat
    if (settings_.xenblockscan_enabled) {
        double t = now_s();
        if (t - last_xbs_holdings_ >= settings_.xenblockscan_holdings_interval_s) {
            last_xbs_holdings_ = t;
            auto bal = wallet_ ? wallet_->current() : std::nullopt;
            auto st = metrics_->stats();
            xbs_.report_holdings(settings_.address, settings_.worker,
                                 bal ? std::optional<double>(bal->xnm) : std::nullopt,
                                 bal ? std::optional<double>(bal->xuni) : std::nullopt,
                                 bal ? std::optional<double>(bal->xblk) : std::nullopt,
                                 st.hps_ema > 0 ? std::optional<double>(st.hps_ema) : std::nullopt,
                                 settings_.tracker_id);
            std::optional<int> diff;
            bool net_ok = false;
            {
                std::lock_guard<std::mutex> lock(state_mu_);
                diff = network_difficulty_;
                net_ok = network_ok_;
            }
            xbs_.report_tracker(settings_.tracker_id, settings_.address, settings_.worker,
                                st.hps_ema > 0 ? std::optional<double>(st.hps_ema) : std::nullopt,
                                st.accepted_total(), st.rejected_total(), st.found_total(),
                                diff, net_ok);
        }
    }
}

void Supervisor::queue_hit(const BlockHit& hit, const std::string& kind, const std::string& reason,
                           const std::string& retry_when) {
    // All block types (XNM, XBLK, XUNI) are first-class in the disk queue.
    if (!store_->enqueue(hit, reason)) {
        return;  // duplicate digest already bagged
    }
    metrics_->record_enqueued(kind);
    metrics_->sync_pending(store_->pending_count());
    // Wake CPU submit worker so it can flush when the pool returns.
    submit_cv_.notify_one();
    if (bag_forward_) bag_forward_->notify();
    ui_event("QUEUED", kind, retry_when);
    log("info", "QUEUED " + kind + " (" + reason + ") — retry when " + retry_when);
}

void Supervisor::enqueue_live_submit(BlockHit hit, std::string kind) {
    {
        std::lock_guard<std::mutex> lock(submit_mu_);
        live_submit_q_.push_back(LiveSubmitJob{std::move(hit), std::move(kind)});
    }
    submit_cv_.notify_one();
}

void Supervisor::handle_batch_hits(MineBatchResult& batch) {
    if (!batch.hits.empty()) {
        for (auto& h : batch.hits) handle_hit(std::move(h));
        return;
    }
    if (batch.hit) handle_hit(std::move(*batch.hit));
}

void Supervisor::handle_hit(BlockHit hit) {
    // Mining-thread path only: prepare + hand off. Never call HTTP here — that is
    // the CPU submit worker so CUDA keeps hashing at full speed.
    if (!is_hex64_key(hit.key)) {
        log("warn", "Dropped GPU hit — key is not 64 hex chars (keygen/CUDA garbage)");
        return;
    }
    std::string kind = classify_block(hit.hash_str, hit.block_type);
    if (kind == "OTHER") kind = hit.block_type.empty() ? "OTHER" : hit.block_type;
    metrics_->record_found(kind);
    log("info", "HIT " + kind + " key=" + hit.key.substr(0, std::min<size_t>(16, hit.key.size())) +
                    "...");
    ui_event("FOUND", kind, hit.strategy);

    auto prepared = prepare_hit_for_submit(hit, settings_.salt_hex(), mining_difficulty(),
                                           settings_.time_cost, settings_.parallelism,
                                           settings_.hash_len);
    if (!prepared) {
        log("info", "Filtered false positive (GPU match did not assemble as Argon2 PHC)");
        return;
    }
    hit = *prepared;
    kind = hit.block_type;

    if (shutting_down_) {
        queue_hit(hit, kind,
                  kind == "XUNI" && !in_xuni_submit_window() ? OUTSIDE_XUNI_WINDOW_REASON
                                                             : SHUTDOWN_PENDING_REASON,
                  "next start");
        return;
    }

    if (kind == "XUNI" && !in_xuni_submit_window()) {
        queue_hit(hit, kind, OUTSIDE_XUNI_WINDOW_REASON, "next XUNI window");
        return;
    }

    // After the first real accept this window: bag new finds so older queue goes first.
    // Before that, live-submit new hits — retrying the same 401'd oldest hashes
    // forever is how a whole m=100 window can land zero accepts.
    if (match_drain_active() && match_drain_flushed_.load() > 0) {
        queue_hit(hit, kind, MATCH_WINDOW_NEW_REASON, "after older queue");
        return;
    }

    {
        const int hit_m = hit.memory_cost.value_or(mining_difficulty());
        if (!can_submit_hit_m(hit_m)) {
            double defer_until = 0;
            bool net_ok = false;
            int net_m = 0;
            {
                std::lock_guard<std::mutex> lock(state_mu_);
                defer_until = defer_submit_until_;
                net_ok = network_ok_;
                net_m = network_difficulty_.value_or(0);
            }
            const char* reason = "network_down";
            const char* retry = "network back";
            if (net_ok && net_m > 0 && net_m != hit_m) {
                reason = DIFFICULTY_CHANGE_REASON;
                retry = "difficulty matches";
            } else if (!force_mine_mode() && now_s() < defer_until) {
                reason = DIFFICULTY_CHANGE_REASON;
                retry = "mining stable";
            }
            queue_hit(hit, kind, reason, retry);
            return;
        }
    }

    // Fast path: hand to CPU submit worker (no HTTP on mining thread).
    enqueue_live_submit(std::move(hit), std::move(kind));
}

void Supervisor::process_live_submit(BlockHit hit, std::string kind) {
    // Runs on CPU submit worker only.
    if (shutting_down_) {
        queue_hit(hit, kind, SHUTDOWN_PENDING_REASON, "next start");
        return;
    }
    if (kind == "XUNI" && !in_xuni_submit_window()) {
        queue_hit(hit, kind, OUTSIDE_XUNI_WINDOW_REASON, "next XUNI window");
        return;
    }
    // Status only — never replan CUDA from this thread.
    refresh_network(false, false);
    const int hit_m = hit.memory_cost.value_or(mining_difficulty());
    if (!can_submit_hit_m(hit_m)) {
        if (live_submit_allowed() && !network_matches_hit_m(hit_m)) {
            queue_hit(hit, kind, DIFFICULTY_CHANGE_REASON, "difficulty matches");
        } else {
            queue_hit(hit, kind, "network_down", "network back");
        }
        return;
    }

    auto result = submitter_->submit(hit, live_submit_timeout_s());
    if (result.ok) {
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            submit_backoff_until_ = 0;
            network_ok_ = true;
        }
        if (match_drain_active() && match_drain_flushed_.fetch_add(1) == 0) {
            log("info", std::string("Match-flush first accept (live hit) — ") +
                            (match_drain_gpu_parked_ ? "CUDA parked" : "CUDA still mining"));
            submit_cv_.notify_all();
        }
        metrics_->record_accepted_live(kind);
        local_stats_->record_accept(kind);
        auto hint = submit_response_hint(result.status, result.body);
        ui_event("ACCEPTED", kind, hint);
        log("info", "SUBMIT OK " + kind + " " + hint + " (CPU submit worker)");
        store_->record_direct_submit(hit, result.status, result.body);
        xbs_.report_accepted(settings_.address, kind, hit.key, hit.hash_str, settings_.worker,
                             hit.memory_cost.value_or(mining_difficulty()));
        return;
    }

    if (result.status == 0 || is_transient_submit_failure(result.status, result.body)) {
        note_submit_transport_failure("Live submit", result.status);
        queue_hit(hit, kind, "network_down", "network back");
        return;
    }
    if (is_pool_hold(result.status, result.body)) {
        queue_hit(hit, kind, DIFFICULTY_CHANGE_REASON, "pool takes it");
        return;
    }
    if (is_difficulty_mismatch(result.status, result.body)) {
        queue_hit(hit, kind, DIFFICULTY_CHANGE_REASON, "difficulty matches");
        return;
    }
    if (is_xuni_window_reject(result.status, result.body)) {
        queue_hit(hit, kind, OUTSIDE_XUNI_WINDOW_REASON, "next XUNI window");
        return;
    }

    const bool parked = is_difficulty_mismatch(result.status, result.body);
    if (store_->record_rejection(hit, result.status, result.body, "live",
                                 parked ? "parked" : "resubmission")) {
        if (counts_as_reject(result.status, result.body)) metrics_->record_rejected_live(kind);
        metrics_->record_resubmission(kind);
        metrics_->sync_pending(store_->pending_count());
    }
    ui_event("RESUBMIT", kind, "HTTP " + std::to_string(result.status));
    log("warn", "Live submit failed " + kind + " — queued for retry");
}

int Supervisor::try_flush_pending(const std::string& context, bool on_shutdown) {
    // CPU submit worker only (or shutdown after mining stopped).
    // Eligibility first — do not copy a 4k+ queue while we are in backoff.
    if (!on_shutdown && !flush_submit_allowed()) {
        log_flush_skip("submit not allowed (backoff or net down)");
        return 0;
    }
    if (!on_shutdown && !force_mine_mode()) {
        std::lock_guard<std::mutex> lock(state_mu_);
        if (now_s() < defer_submit_until_) return 0;
    }
    // Never replan CUDA from the submit worker; on shutdown mining is already stopped.
    const bool refreshed = refresh_network(on_shutdown, /*replan_engine=*/false);
    refresh_paper();
    if (!refreshed) {
        if (on_shutdown) {
            store_->defer_all_to_next_start();
            return 0;
        }
        // Hybrid: a flaky /difficulty poll used to abort the whole match window
        // (flushed~0 with thousands still queued). Last-good m= is enough to send.
        if (!flush_submit_allowed() || !network_matches_hit_m(bag_target_m())) {
            log_flush_skip("difficulty poll failed and last-good does not match bag");
            return 0;
        }
    }
    if (!on_shutdown && !flush_submit_allowed()) {
        log_flush_skip("submit not allowed after difficulty refresh");
        return 0;
    }

    int net_m = submit_target_m();
    if (net_m <= 0) {
        log_flush_skip("no last-good network/paper m=");
        return 0;
    }

    const bool draining = match_drain_active_.load();
    int wave = 64;
    if (on_shutdown) {
        wave = 0;  // whole matching bag
    } else if (draining) {
        const int par = std::max(1, std::min(256, settings_.match_drain_parallel));
        wave = std::max(256, par * 16);
        if (settings_.match_drain_batch > 0) wave = std::min(wave, settings_.match_drain_batch);
    }
    const int fetch = on_shutdown ? 0 : wave * 2;
    auto pending = store_->list_flush_batch(net_m, bag_target_m(), fetch, flush_skip_before_id_);
    if (pending.empty()) return 0;

    // Build work list eligible for submit (CPU-only path; GPU never blocked here).
    struct WorkItem {
        int64_t id = 0;
        BlockHit hit;
    };
    std::vector<WorkItem> work;
    work.reserve(pending.size());
    for (const auto& pb : pending) {
        auto ready = ready_to_flush(pb.hit.block_type);
        if (!ready.first) continue;
        auto hit = pb.hit;
        const int hit_m = hit.memory_cost.value_or(mining_difficulty());
        // Eligibility mirrors list_flush_batch: submit hit m >= net m only
        // (server rejects strictly-lower m; higher-m parked finds flush when diff falls).
        if (hit_m < net_m) continue;
        if (!is_argon2_encoded(hit.hash_str)) {
            auto prepared = prepare_hit_for_submit(hit, settings_.salt_hex(), hit_m,
                                                   settings_.time_cost, settings_.parallelism,
                                                   settings_.hash_len);
            if (!prepared) {
                store_->mark_submitted(pb.id, 0, "prepare failed");
                continue;
            }
            hit = *prepared;
        }
        work.push_back(WorkItem{pb.id, std::move(hit)});
        if (!on_shutdown && wave > 0 && static_cast<int>(work.size()) >= wave) break;
    }
    if (work.empty()) {
        if (!pending.empty()) flush_skip_before_id_ = pending.back().id + 1;
        log_flush_skip("0 eligible of " + std::to_string(store_->pending_count()) +
                       " pending (m= or XUNI window)");
        return 0;
    }

    const int timeout_s = live_submit_timeout_s();
    int parallel = on_shutdown ? 1 : 4;
    if (draining && !on_shutdown) {
        parallel = std::max(1, std::min(256, settings_.match_drain_parallel));
    }
    if (draining && !match_drain_logged_first_wave_) {
        match_drain_logged_first_wave_ = true;
        log("info", std::string("Match-flush sending /verify wave of ") +
                        std::to_string(work.size()) + " (parallel=" + std::to_string(parallel) +
                        ", bag~" + std::to_string(store_->pending_count()) + ")");
    }

    int flushed = 0;
    int holds = 0;
    int last_hold_status = 0;
    std::string last_hold_hint;
    bool transport_fail = false;
    for (size_t base = 0; base < work.size(); base += static_cast<size_t>(parallel)) {
        if (transport_fail) break;  // (KIMI) break AFTER finishing current wave
        const size_t n = std::min(static_cast<size_t>(parallel), work.size() - base);
        std::vector<std::future<SubmitResult>> futs;
        futs.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            BlockHit hit = work[base + i].hit;
            futs.push_back(std::async(std::launch::async, [this, hit, timeout_s]() {
                return submitter_->submit(hit, timeout_s);
            }));
        }
        std::vector<int64_t> accepted_ids;
        accepted_ids.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            auto result = futs[i].get();
            auto& item = work[base + i];
            if (result.ok) {
                accepted_ids.push_back(item.id);
                metrics_->record_accepted_flush(item.hit.block_type);
                local_stats_->record_accept(item.hit.block_type);
                xbs_.report_accepted(settings_.address, item.hit.block_type, item.hit.key,
                                     item.hit.hash_str, settings_.worker,
                                     item.hit.memory_cost.value_or(mining_difficulty()));
                ++flushed;
                if (draining) match_drain_flushed_.fetch_add(1);
                std::lock_guard<std::mutex> lock(state_mu_);
                submit_backoff_until_ = 0;
                network_ok_ = true;
            } else if (result.status == 0 ||
                       is_transient_submit_failure(result.status, result.body)) {
                if (!on_shutdown) {
                    note_submit_transport_failure("Queue flush", result.status);
                    transport_fail = true;
                    store_->mark_pending_reason(item.id, "network_down");  // (KIMI) don't lose it
                }
            } else if (is_pool_hold(result.status, result.body) ||
                       is_difficulty_mismatch(result.status, result.body) ||
                       is_xuni_window_reject(result.status, result.body)) {
                ++holds;
                last_hold_status = result.status;
                last_hold_hint = submit_response_hint(result.status, result.body);
                // 401 difficulty: park the block — it becomes submittable again
                // automatically when network difficulty falls to/below its m
                // (eligibility filter hit_m >= net_m keeps it out of flush waves until then).
                if (is_difficulty_mismatch(result.status, result.body)) {
                    store_->mark_pending_reason(item.id, "parked");
                }
            } else {
                if (counts_as_reject(result.status, result.body)) {
                    metrics_->record_rejected_flush(item.hit.block_type);
                }
                store_->log_rejection(item.hit, result.status, result.body, "flush");
                store_->mark_pending_reason(item.id, "resubmission");
            }
        }
        if (!accepted_ids.empty()) store_->mark_submitted_many(accepted_ids);
    }
    metrics_->sync_pending(store_->pending_count());
    if (flushed) {
        if (draining && match_drain_flushed_.load() == flushed) {
            log("info", std::string("Match-flush first accept — ") +
                            (match_drain_gpu_parked_ ? "CUDA parked" : "CUDA still mining"));
        }
        log("info", "Submitted " + std::to_string(flushed) + " queued block(s) (" + context +
                        ", CPU submit x" + std::to_string(parallel) + ")");
        ui_event("ACCEPTED", "QUEUE", context + " flushed " + std::to_string(flushed));
        submit_cv_.notify_all();
    } else if (!on_shutdown && holds > 0 && !transport_fail) {
        // Instant 401s used to retry at ~16 req/s and WAF the window. Pause, keep bag.
        const double pause = draining ? 1.0 : 2.0;
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            submit_backoff_until_ = now_s() + pause;
        }
        if (!work.empty()) {
            flush_skip_before_id_ = work.back().id + 1;
        }
        const double t = now_s();
        if (t - last_flush_skip_log_at_ >= 10.0) {
            last_flush_skip_log_at_ = t;
            log("warn", std::string(draining ? "Match-flush" : "Queue flush") +
                            " /verify hold HTTP " + std::to_string(last_hold_status) + " x" +
                            std::to_string(holds) + " — " + last_hold_hint + " — pause " +
                            std::to_string(static_cast<int>(pause)) + "s, rotate past id " +
                            std::to_string(flush_skip_before_id_) + ", bag kept");
        }
    }
    return flushed;
}

void Supervisor::service_pending_queue(double now) {
    // CPU submit worker only — never touches the GPU mining loop.
    const bool submit_win = in_xuni_submit_window();
    const bool taper = in_xuni_taper_window();
    auto counts = store_->pending_by_type();
    int pending_xuni = counts["XUNI"];
    int pending_total = store_->pending_count();

    // :55 — start flushing XUNI one minute before mine window opens.
    if (submit_win && !was_in_xuni_window_) {
        was_in_xuni_window_ = true;
        if (pending_xuni) {
            log("info", "XUNI submit window open (:55–:04) — flushing queued XUNI (CPU worker)");
            try_flush_pending("XUNI submit window");
            last_xuni_flush_ = now;
            last_queue_flush_ = now;
        }
    } else if (!submit_win) {
        was_in_xuni_window_ = false;
    }

    // Match-flush: launch the next HTTP wave immediately after the last one returns.
    const bool draining = match_drain_active_.load();
    double flush_every = pending_total > 500 ? 1.0 : (pending_total > 50 ? 2.0 : 5.0);
    if (draining) flush_every = 0.0;
    if (taper && pending_xuni > 0) flush_every = std::min(flush_every, 0.25);
    if (now - last_queue_flush_ >= flush_every) {
        try_flush_pending(draining ? "match-flush" : "queue service");
        last_queue_flush_ = now;
    } else if (submit_win && pending_xuni) {
        const double xuni_every = taper ? 0.5 : 2.0;
        if (now - last_xuni_flush_ >= xuni_every) {
            try_flush_pending(taper ? "XUNI taper" : "XUNI interval");
            last_xuni_flush_ = now;
        }
    }
}

void Supervisor::start_submit_worker() {
    if (submit_worker_running_) return;
    submit_worker_running_ = true;
    submit_thread_ = std::thread([this] { submit_worker_loop(); });
    log("info", "CPU submit worker started (HTTP submit/flush off mining thread)");
}

void Supervisor::stop_submit_worker() {
    if (!submit_worker_running_ && !submit_thread_.joinable()) return;
    submit_worker_running_ = false;
    submit_cv_.notify_all();
    if (submit_thread_.joinable()) submit_thread_.join();
}

void Supervisor::submit_worker_loop() {
    while (true) {
        std::deque<LiveSubmitJob> batch;
        {
            std::unique_lock<std::mutex> lock(submit_mu_);
            // Wake often so large queues drain on the CPU path without waiting on the GPU.
            submit_cv_.wait_for(lock, std::chrono::milliseconds(250), [this] {
                return !live_submit_q_.empty() || !submit_worker_running_ || shutting_down_.load();
            });
            batch.swap(live_submit_q_);
        }

        // While shutting down, never open new HTTP — bag to disk instead.
        if (shutting_down_.load()) {
            for (auto& job : batch) {
                store_->enqueue(job.hit, SHUTDOWN_PENDING_REASON);
                metrics_->record_enqueued(job.kind);
            }
            if (!batch.empty()) metrics_->sync_pending(store_->pending_count());
        } else {
            for (auto& job : batch) {
                process_live_submit(std::move(job.hit), std::move(job.kind));
            }
            service_pending_queue(now_s());
        }

        if (!submit_worker_running_) {
            // Final bag of anything still in the live queue (no HTTP hang on close).
            int bagged = bag_live_queue_to_store(SHUTDOWN_PENDING_REASON);
            if (store_) store_->defer_all_to_next_start();
            // Best-effort submit only if network already healthy — never block close on timeouts.
            if (!shutting_down_.load() || live_submit_allowed()) {
                try_flush_pending("shutdown", true);
            } else if (bagged || (store_ && store_->pending_count() > 0)) {
                log("info", "Shutdown: kept " + std::to_string(store_->pending_count()) +
                                " block(s) on disk for next start");
            }
            break;
        }
    }
    log("info", "CPU submit worker stopped");
}

bool Supervisor::gpu_safety_tick(const GpuSnapshot* snap) {
    if (!snap) return true;
    ThermalHuntResult hunt;
    if (engine_) {
        hunt = engine_->update_thermal_batch_from_temp(snap->temperature_c, snap->memory_junction_c,
                                                       now_s());
        static int last_logged_batch = -1;
        if (hunt.action && std::strcmp(hunt.action, "hold") != 0 && hunt.batch != last_logged_batch) {
            last_logged_batch = hunt.batch;
            std::string sensor = (hunt.sensor && std::strcmp(hunt.sensor, "mem") == 0)
                                     ? "mem junc"
                                     : "GPU";
            log("info", std::string("Thermal ") + hunt.action + " " +
                            std::to_string(settings_.thermal_batch_step) + " — " + sensor + " " +
                            std::to_string(hunt.control_c) + "C (target " +
                            std::to_string(settings_.warn_mem_temp_c) + "C / cap " +
                            std::to_string(settings_.max_mem_temp_c) + "C) batch " +
                            std::to_string(hunt.batch) + "/lane idle " +
                            std::to_string(hunt.idle_ms) + "ms");
        }
        if (hunt.lane_delta != 0) {
            // 4x2 pair queue needs all 8 lanes. Ignore hunt lane steps.
        }
    }
    if (power_) power_->tick(snap);
    if (vram_caps_ && snap->used_mib >= vram_caps_->emergency_mib) {
        double scale = engine_->note_vram_emergency();
        log("warn", "VRAM emergency at " + std::to_string(snap->used_mib) + "MiB (cap " +
                        std::to_string(vram_caps_->emergency_mib) +
                        "MiB) — shrinking next pack to " +
                        std::to_string(static_cast<int>(scale * 100.0)) + "%");
        try {
            engine_->stop();
        } catch (...) {
        }
        // Short pause so NVML drops; do not use the 60s thermal cooldown.
        cooldown_until_ = now_s() + 3.0;
        return false;
    }
    return true;
}

void Supervisor::graceful_shutdown(const std::string& reason) {
    if (shutting_down_) return;
    shutting_down_ = true;
    running_ = false;
    log("info", reason);
    if (dashboard_) dashboard_->set_status("Stopping — bagging queue...");
    try {
        // Finalize double-buffered GPU results before tearing down CUDA.
        if (engine_ && engine_->is_running()) {
            auto drained = engine_->drain_pipeline();
            handle_batch_hits(drained);
            if (drained.hashes_done > 0) {
                metrics_->record_hashes(drained.hashes_done, engine_->last_hashrate());
            }
        }
        if (engine_) engine_->stop();
    } catch (...) {
    }
    // Bag in-flight live submits to disk first (fast), then stop worker.
    bag_live_queue_to_store(SHUTDOWN_PENDING_REASON);
    if (store_) {
        store_->defer_all_to_next_start();
        store_->flush();
    }
    if (bag_forward_) bag_forward_->stop();
    stop_submit_worker();
}

void Supervisor::finalize_session() {
    if (finalized_) return;
    finalized_ = true;
    running_ = false;
    shutting_down_ = true;
    // Ensure anything still only in RAM is on disk before we tear down.
    bag_live_queue_to_store(SHUTDOWN_PENDING_REASON);
    if (store_) store_->defer_all_to_next_start();
    if (bag_forward_) bag_forward_->stop();
    stop_submit_worker();
    if (woody_) woody_->stop();
    if (timelapse_) timelapse_->finalize();
    try {
        if (engine_) engine_->stop();
    } catch (...) {
    }
    if (power_) power_->restore();
    if (paper_) paper_->stop();
    if (poller_) poller_->stop();
    xbs_.stop();
    if (gpu_) gpu_->shutdown();
    if (lock_) lock_->release();
    if (dashboard_) {
        dashboard_->set_status("Stopped");
        dashboard_->stop();
    }
    log("info", "=== XNMiner CUDA stopped ===");
}

void Supervisor::run(std::optional<int> max_seconds) {
    session_started_at_ = now_s();
    running_ = true;
    shutting_down_ = false;

#ifdef _WIN32
    if (SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS)) {
        log("info", "Process priority set to Above Normal");
    }
#endif

    if (dashboard_) {
        dashboard_->start();
        dashboard_->set_status("Starting CUDA engine...");
        dashboard_->render();
    }

    poller_->start();
    if (paper_) paper_->start();
    // Initial network probe — short timeouts so a dead pool cannot delay mining start.
    // Mining uses last-good / fallback difficulty; submits queue until the pool returns.
    for (int i = 0; i < 3 && running_; ++i) {
        NetworkStatus st = poller_->poll_once(std::min(settings_.connection_timeout_s, 5));
        if (st.difficulty) {
            refresh_network(false, /*replan_engine=*/false);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    // Absorb whatever the background poller has (if anything).
    refresh_network(false, /*replan_engine=*/false);
    {
        std::lock_guard<std::mutex> lock(state_mu_);
        if (network_difficulty_) {
            log("info", "Network difficulty: " + std::to_string(*network_difficulty_) +
                            (network_ok_ ? "" : " (submit deferred — mining anyway)"));
        } else {
            log("warn", "Network difficulty unavailable — using fallback for net display m=" +
                            std::to_string(settings_.memory_cost));
            network_difficulty_ = settings_.memory_cost;
            network_ok_ = false;
        }
    }
    if (force_mine_mode()) {
        log("info", "HYBRID mode: CUDA mine m=" + std::to_string(forced_mine_m()) +
                        " fixed; flush when lastblock newest m= OR /difficulty matches" +
                        (settings_.match_drain_max_s > 0
                             ? (" (match_flush max " +
                                std::to_string(settings_.match_drain_max_s) + "s)")
                             : " (until bag empty or both oracles leave)"));
        if (dashboard_) {
            dashboard_->set_mining_m(forced_mine_m(), true);
            dashboard_->set_status("Hybrid: mining fixed m=" + std::to_string(forced_mine_m()));
        }
    }
    if (settings_.xuni_mining_enabled) {
        log("info", std::string("XUNI enabled — ") + XUNI_WINDOW_LABEL);
    } else {
        log("info", "XUNI mining OFF — GPU hunts XNM/XBLK only; queued XUNI still flush " +
                        std::string(XUNI_WINDOW_LABEL));
    }

    try {
        apply_vram_policy();
        engine_->start();
        apply_vram_policy();
        // Hybrid: always lock CUDA to force m=. Classic: follow last network m=.
        if (force_mine_mode()) {
            engine_->set_difficulty(forced_mine_m());
        } else {
            std::lock_guard<std::mutex> lock(state_mu_);
            if (network_difficulty_) engine_->set_difficulty(*network_difficulty_);
        }
        if (auto* plan = engine_->vram_plan()) {
            log("info", "CUDA VRAM plan: " + plan->summary());
        }
        log("info", "CUDA device: " + engine_->device_name());
        log("info", "CPU: " + std::to_string(settings_.cpu_physical) + " cores / " +
                        std::to_string(settings_.cpu_logical) + " threads; keygen " +
                        std::to_string(settings_.keygen_threads) +
                        " workers (not oversubscribed); flush parallel " +
                        std::to_string(settings_.match_drain_parallel));
        if (engine_) {
            const int lanes = engine_->active_lanes();
            log("info", "CUDA lanes: " + std::to_string(lanes) +
                            " (cap " + std::to_string(settings_.cuda_max_lanes) +
                            ") keep-hot pipeline, VRAM cap " +
                            std::to_string(static_cast<int>(settings_.target_vram_pct)) + "%");
        }
        if (auto tsnap = gpu_->snapshot()) {
            std::string temps = "GPU die " + std::to_string(tsnap->temperature_c) + "C";
            if (tsnap->has_memory_junction()) {
                temps += ", memory junction " + std::to_string(tsnap->memory_junction_c) + "C";
                if (settings_.thermal_use_memory_junction) {
                    temps += " (hold " + std::to_string(settings_.warn_mem_temp_c) +
                             "C / cap " + std::to_string(settings_.max_mem_temp_c) +
                             "C, start " +
                             std::to_string(static_cast<int>(settings_.gpu_thermal_start_scale * 100.0)) +
                             "% pack, 8 lanes fixed)";
                } else {
                    temps += " (display only — thermal_use_memory_junction=false)";
                }
            } else {
                temps += ", memory junction unavailable (NVML N/A, NVAPI failed)";
            }
            log("info", "Temps: " + temps);
        }
    } catch (const std::exception& ex) {
        log("error", std::string("CUDA start failed: ") + ex.what());
        std::cerr << "\nERROR: CUDA start failed: " << ex.what() << "\n"
                  << "If the message mentions driver/runtime versions: your host GPU driver is\n"
                  << "older than the CUDA toolkit this binary was built with. Rebuild with a\n"
                  << "matching toolkit:  rm -rf build data/cuda_arch && ./start-miner.sh\n"
                  << "(the build scripts pick a toolkit compatible with your driver).\n\n";
        finalize_session();
        return;
    }

    if (power_) power_->apply();

    if (settings_.woodyminer_enabled && !settings_.address.empty()) {
        woody_ = std::make_unique<WoodyminerUploader>(
            settings_.woodyminer_upload_url, settings_.woodyminer_upload_period_s,
            settings_.woodyminer_custom_name, settings_.address,
            derive_machine_id(settings_.device_id),
            [this] { return metrics_->stats(); },
            [this] { return gpu_->snapshot(); },
            // Public: live network m= only (or 0 = N/A). Never hybrid mine m=.
            [this] { return network_difficulty_for_public(); }, session_started_at_, logger_.get());
        woody_->start();
    }

    // CPU worker owns all HTTP submit + queue flush (startup flush included).
    start_submit_worker();
    if (!settings_.bag_forward_url.empty()) {
        bag_forward_ = std::make_unique<BagForwarder>(
            *store_, settings_.bag_forward_url, settings_.bag_forward_token, settings_.worker,
            settings_.bag_forward_batch, logger_.get());
        bag_forward_->start();
    }
    if (store_->pending_count() > 0) {
        log("info", "Resuming with " + std::to_string(store_->pending_count()) +
                        " queued block(s) — CPU submit worker will flush");
        submit_cv_.notify_one();
    }

    log("info", "=== Mining started (pure CUDA) ===");
    if (dashboard_) dashboard_->set_status("Mining");

    // Mining timer starts after engine is up (not during VRAM alloc / probes).
    const double mining_started_at = now_s();
    session_started_at_ = mining_started_at;
    double end_at = max_seconds ? mining_started_at + *max_seconds : 0;
    last_queue_flush_ = now_s();
    last_ui_ = 0;

    while (running_) {
        if (end_at > 0 && now_s() >= end_at) {
            graceful_shutdown("Max seconds reached");
            break;
        }

        double t = now_s();
        // Mining thread: network status + CUDA replan only. No HTTP submit here.
        refresh_network(false, /*replan_engine=*/true);
        refresh_paper();
        // CPU flush when paper/thermometer shows bag m=. Park CUDA only on live match.
        update_match_drain(t);
        maybe_check_update(t);
        if (match_drain_active()) submit_cv_.notify_all();

        auto snap = gpu_->snapshot();
        if (t < cooldown_until_) {
            if (dashboard_) dashboard_->set_status("GPU cooling down...");
            if (t - last_ui_ >= settings_.stats_interval_s) {
                last_ui_ = t;
                ui_refresh();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            // Restart after cooldown
            if (now_s() >= cooldown_until_ && !engine_->is_running()) {
                if (reduce_lanes_after_cooldown_) {
                    int after = engine_->reduce_lane_cap();
                    log("warn", "Lane cap reduced after temp cooldown to " + std::to_string(after));
                    reduce_lanes_after_cooldown_ = false;
                }
                try {
                    if (auto idle = gpu_->snapshot()) {
                        engine_->set_occupied_vram_mib(idle->used_mib);
                    }
                    engine_->start();
                    if (force_mine_mode()) {
                        engine_->set_difficulty(forced_mine_m());
                    } else {
                        std::optional<int> diff;
                        {
                            std::lock_guard<std::mutex> lock(state_mu_);
                            diff = network_difficulty_;
                        }
                        if (diff) engine_->set_difficulty(*diff);
                    }
                    if (auto* plan = engine_->vram_plan()) {
                        log("info", "CUDA VRAM plan: " + plan->summary());
                    }
                    if (dashboard_) dashboard_->set_status("Mining");
                } catch (const std::exception& ex) {
                    log("error", std::string("CUDA restart failed: ") + ex.what());
                    cooldown_until_ = now_s() + settings_.gpu_cooldown_s;
                }
            }
            continue;
        }

        if (!gpu_safety_tick(snap ? &*snap : nullptr)) {
            continue;
        }

        try {
            if (!engine_->is_running()) engine_->start();
            // Park CUDA only when live /difficulty is bag m=. Mixed windows keep mining.
            if (should_park_cuda()) {
                if (!match_drain_gpu_parked_ && engine_->is_running()) {
                    auto leftover = engine_->drain_pipeline();
                    if (leftover.hashes_done > 0) {
                        metrics_->record_hashes(leftover.hashes_done, engine_->last_hashrate());
                    }
                    handle_batch_hits(leftover);
                    match_drain_gpu_parked_ = true;
                    log("info", "CUDA parked for match-flush — live m= matches bag, CPU owns /verify");
                }
                if (t - last_ui_ >= settings_.stats_interval_s) {
                    last_ui_ = t;
                    ui_refresh();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            if (match_drain_gpu_parked_) {
                match_drain_gpu_parked_ = false;
                log("info", "CUDA resumed mining — flush continues on CPU (live m= left bag)");
            }
            // Queue-type scan only matters inside the XUNI mine window (soft-cap / throttle).
            // Outside that window, skip the O(queue) walk so the GPU path stays hot.
            if (settings_.xuni_mining_enabled && in_xuni_window()) {
                auto counts = store_->pending_by_type(false);
                engine_->set_queue_counts(counts["XUNI"], counts["XNM"], counts["XBLK"]);
            } else {
                engine_->set_queue_counts(0, 0, 0);
            }
            static bool logged_xuni_pause = false;
            auto batch = engine_->mine_batch();
            if (!engine_->xuni_mining_active()) {
                if (!logged_xuni_pause && in_xuni_window()) {
                    log("info", "XUNI mining paused/throttled — focusing XEN11 (XNM/XBLK); "
                                "soft-cap, 1-in-N, or taper minute");
                    logged_xuni_pause = true;
                }
            } else {
                logged_xuni_pause = false;
            }
            metrics_->set_active_lanes(engine_->active_lanes());
            if (batch.hashes_done > 0) {
                metrics_->record_hashes(batch.hashes_done, engine_->last_hashrate());
            }
            handle_batch_hits(batch);
            const int idle_ms = engine_->thermal_idle_ms();
            if (idle_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));
            }
        } catch (const std::exception& ex) {
            log("error", std::string("Mine batch error: ") + ex.what());
            const std::string msg = ex.what();
            if (msg.find("illegal memory access") != std::string::npos ||
                msg.find("CUDA error 700") != std::string::npos) {
                log("error", "CUDA 700 — resetting device and restarting engine");
                try { engine_->stop(); } catch (...) {}
                cudaDeviceReset();
                cooldown_until_ = now_s() + 3.0;
            } else {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }

        if (t - last_ui_ >= settings_.stats_interval_s) {
            last_ui_ = t;
            ui_refresh();
            if (!dashboard_) {
                auto st = metrics_->stats();
                const int up_s = session_started_at_ > 0
                                     ? static_cast<int>(std::max(0.0, now_s() - session_started_at_))
                                     : 0;
                const double avg_xnm = st.avg_xnm_per_s(up_s);
                std::ostringstream avg_oss;
                avg_oss << std::fixed << std::setprecision(avg_xnm >= 1.0 ? 2 : 4) << avg_xnm;
                log("info", "H/s=" + std::to_string(static_cast<int>(st.hps_ema)) +
                                " kernH/s=" +
                                std::to_string(static_cast<int>(engine_->last_kernel_hashrate())) +
                                " lanes=" + std::to_string(engine_->active_lanes()) +
                                " batch=" + std::to_string(engine_->batch_size()) +
                                " avgXNM/s=" + avg_oss.str() +
                                " found=" + std::to_string(st.found_total()) +
                                " accept=" + std::to_string(st.accepted_total()) +
                                " queue=" + std::to_string(store_->pending_count()));
            }
        }
    }

    if (!shutting_down_) graceful_shutdown("Stop requested");
    finalize_session();
}

}  // namespace xn
