#pragma once

#include <filesystem>
#include <string>

namespace xn {

struct Settings {
    std::string address;
    std::string worker;
    std::string base_url = "http://xenblocks.io";
    int connection_timeout_s = 12;
    int network_poll_interval_s = 6;
    // /difficulty is flaky; fail fast and retry so short m=100 windows are not missed.
    int network_poll_timeout_s = 8;
    int network_down_poll_interval_s = 8;

    // Newspaper oracle (sealed-block m=). Faster/healthier than /difficulty.
    std::string lastblock_url = "http://xenblocks.io:4445/getblocks/lastblock";
    std::string lastblock_url_fallback = "http://xenblocks.io:4447/getblocks/lastblock";
    int lastblock_poll_interval_s = 2;
    int lastblock_timeout_s = 8;

    std::string strategy = "random";
    int memory_cost = 1100;
    int time_cost = 1;
    int parallelism = 1;
    int hash_len = 64;

    // Hybrid / force-mine: if > 0, CUDA always uses this Argon2 m= (100 = harvest).
    // Network /difficulty is still polled; hits are queued until RPC m= matches, then flushed.
    // 0 = classic mode (mine whatever the network reports).
    int force_mine_memory_cost = 100;

    // CPU-flush while paper newest m= or /difficulty matches bag.
    // Park CUDA only when live /difficulty is bag m=. Always mine otherwise.
    bool match_drain_enabled = true;
    // Enter flush mode when at least this many pending blocks match current net m=.
    int match_drain_min_queue = 1;
    // Max seconds for one flush window. 0 = stay until bag empty or both oracles leave.
    int match_drain_max_s = 0;
    // Parallel /verify workers during flush.
    int match_drain_parallel = 0;
    // Cap submits per flush wave (0 = no cap / whole matching bag).
    int match_drain_batch = 0;

    // Value priority: XNM > XBLK, plus XUNI hunting in the :56–:04 window (fleet
    // evidence 2026-08-29: XUNI is the dominant accepted-count stream — off = regres
    // vs the old miner). Cost is near-zero (pattern scan on the same Argon2 hashes;
    // no extra VRAM, per-lane flag only). Existing queued XUNI flush in the :55–:04 window.
    bool xuni_mining_enabled = true;
    // Pause new XUNI mining when pending XUNI in queue reaches this (hysteresis below).
    int xuni_queue_soft_cap = 100;
    int xuni_queue_resume = 40;
    // When under the cap and inside the XUNI mine window, only 1-in-N batches enable XUNI.
    int xuni_every_n_batches = 3;
    // At most this many CUDA lanes may hunt XUNI (rest stay XEN11-only).
    int xuni_max_lanes = 2;

    double target_vram_pct = 79.0;
    double desktop_headroom_pct = 16.0;
    double emergency_vram_pct = 93.0;
    double min_headroom_pct = 4.0;
    double runtime_overhead_pct = 5.0;
    int min_headroom_floor_mib = 512;
    int runtime_overhead_floor_mib = 256;
    int target_vram_mib = 0;
    int headroom_mib = 0;
    int emergency_vram_mib = 0;
    int min_headroom_mib = 0;

    int max_gpu_temp_c = 84;
    int warn_gpu_temp_c = 78;
    // Memory junction is the primary thermal cap. Hunt batch around warn, never exceed max.
    int max_mem_temp_c = 80;
    int warn_mem_temp_c = 76;
    bool thermal_use_memory_junction = true;
    int gpu_cooldown_s = 15;
    bool gpu_power_boost_enabled = true;
    int gpu_power_target_pct = 100;
    int gpu_power_min_pct = 70;
    bool gpu_difficulty_power_enabled = true;
    double gpu_difficulty_power_full_ratio = 2.0;
    bool gpu_thermal_batch_enabled = true;
    double gpu_thermal_batch_min_scale = 0.30;
    /// First job count as a fraction of the VRAM pack. 0.86 ≈ 24.8k of 28.8k (80C spot).
    double gpu_thermal_start_scale = 0.70;
    int thermal_batch_step = 1;
    int thermal_settle_s = 15;
    bool gpu_windows_performance_mode = false;
    int sample_interval_s = 5;

    std::filesystem::path db_path;
    std::filesystem::path jsonl_path;
    std::filesystem::path rejected_jsonl_path;
    double submit_cpu_fraction = 0.30;
    // Copy every queued hit to the Windows home vault. Empty = local bag only.
    std::string bag_forward_url;
    std::string bag_forward_token;
    int bag_forward_batch = 32;

    std::filesystem::path log_path;
    std::filesystem::path timelapse_path;
    int stats_interval_s = 4;
    int timelapse_sample_s = 30;
    bool dashboard_enabled = true;

    bool woodyminer_enabled = true;
    std::string woodyminer_upload_url = "https://woodyminer.com/api/stat/upload";
    int woodyminer_upload_period_s = 60;
    std::string woodyminer_custom_name;

    bool xenblockscan_enabled = false;
    std::string xenblockscan_endpoint = "http://127.0.0.1:8787/api/v1/events";
    std::string xenblockscan_api_key;
    bool xenblockscan_report_rejects = false;
    int xenblockscan_holdings_interval_s = 30;
    bool xenblockscan_backfill = false;
    std::string tracker_id;

    int device_id = 0;
    int cuda_batch_size = 0;
    int cuda_max_batch_size = 0;
    int cuda_runtime_overhead_mib = 0;
    int vram_reference_difficulty = 1100;
    int cuda_max_lanes = 8;
    int cuda_lane_reserve = 1;
    /// Shared keygen pool size. 0 in miner.ini = auto from CPU cores (never oversubscribe).
    int keygen_threads = 0;
    int cpu_logical = 0;
    int cpu_physical = 0;
    /// VRAM work patches. 2 = half working / half preloaded. 3 = later trifecta.
    /// All lanes hash one patch; the others only store the next job(s).
    int work_patches = 2;

    bool update_check_enabled = false;
    std::string update_github_repo;
    std::string update_github_ref = "main";
    std::string update_token;
    int update_check_interval_s = 300;
    std::filesystem::path update_sha_path;

    std::filesystem::path root;

    std::string salt_hex() const {
        if (address.size() > 2 && (address[0] == '0') && (address[1] == 'x' || address[1] == 'X')) {
            return address.substr(2);
        }
        return address;
    }

    std::string difficulty_url() const {
        auto base = base_url;
        while (!base.empty() && (base.back() == '/' || base.back() == '\\')) base.pop_back();
        return base + "/difficulty";
    }

    std::string verify_url() const {
        auto base = base_url;
        while (!base.empty() && (base.back() == '/' || base.back() == '\\')) base.pop_back();
        return base + "/verify";
    }
};

Settings load_settings(const std::filesystem::path& ini_path);
bool ensure_wallet_configured(const std::filesystem::path& ini_path, bool interactive);
void set_ini_value(const std::filesystem::path& ini_path, const std::string& section,
                   const std::string& key, const std::string& value);

}  // namespace xn
