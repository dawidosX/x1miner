#include "config/settings.hpp"

#include "util/cpu.hpp"
#include "util/paths.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <vector>

namespace xn {
namespace {

std::string trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool parse_bool(const std::string& v, bool def) {
    auto s = to_lower(trim(v));
    if (s.empty()) return def;
    if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
    if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    return def;
}

using Section = std::map<std::string, std::string>;
using Ini = std::map<std::string, Section>;

Ini parse_ini(const std::filesystem::path& path) {
    Ini ini;
    std::ifstream in(path);
    if (!in) return ini;
    std::string line;
    std::string section;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto t = trim(line);
        if (t.empty() || t[0] == '#' || t[0] == ';') continue;
        if (t.front() == '[' && t.back() == ']') {
            section = to_lower(trim(t.substr(1, t.size() - 2)));
            continue;
        }
        auto eq = t.find('=');
        if (eq == std::string::npos) continue;
        auto key = to_lower(trim(t.substr(0, eq)));
        auto val = trim(t.substr(eq + 1));
        // strip inline comments only when preceded by space
        auto hash = val.find(" #");
        if (hash != std::string::npos) val = trim(val.substr(0, hash));
        ini[section][key] = val;
    }
    return ini;
}

std::string get(const Ini& ini, const std::string& sec, const std::string& key,
                const std::string& def = {}) {
    auto sit = ini.find(sec);
    if (sit == ini.end()) return def;
    auto kit = sit->second.find(key);
    if (kit == sit->second.end()) return def;
    return kit->second.empty() ? def : kit->second;
}

int get_i(const Ini& ini, const std::string& sec, const std::string& key, int def) {
    auto s = get(ini, sec, key);
    if (s.empty()) return def;
    try {
        return std::stoi(s);
    } catch (...) {
        return def;
    }
}

double get_d(const Ini& ini, const std::string& sec, const std::string& key, double def) {
    auto s = get(ini, sec, key);
    if (s.empty()) return def;
    try {
        return std::stod(s);
    } catch (...) {
        return def;
    }
}

bool get_b(const Ini& ini, const std::string& sec, const std::string& key, bool def) {
    return parse_bool(get(ini, sec, key), def);
}

bool valid_eth_address(const std::string& a) {
    if (a.size() != 42) return false;
    if (!(a[0] == '0' && (a[1] == 'x' || a[1] == 'X'))) return false;
    for (size_t i = 2; i < a.size(); ++i) {
        if (!std::isxdigit(static_cast<unsigned char>(a[i]))) return false;
    }
    return true;
}

std::string random_worker_name() {
    static constexpr char hex[] = "0123456789abcdef";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 15);
    std::string out = "xnminer-";
    for (int i = 0; i < 8; ++i) out.push_back(hex[dist(gen)]);
    return out;
}

std::string random_tracker_id() {
    static constexpr char hex[] = "0123456789abcdef";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 15);
    std::string out = "xbs-";
    for (int i = 0; i < 16; ++i) out.push_back(hex[dist(gen)]);
    return out;
}

}  // namespace

void set_ini_value(const std::filesystem::path& ini_path, const std::string& section,
                   const std::string& key, const std::string& value) {
    std::vector<std::string> lines;
    {
        std::ifstream in(ini_path);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            lines.push_back(line);
        }
    }

    const std::string sec_hdr = "[" + section + "]";
    const std::string sec_l = to_lower(section);
    const std::string key_l = to_lower(key);
    bool in_sec = false;
    bool wrote = false;
    bool found_sec = false;
    std::vector<std::string> out;

    for (const auto& line : lines) {
        auto t = trim(line);
        if (!t.empty() && t.front() == '[' && t.back() == ']') {
            if (in_sec && !wrote) {
                out.push_back(key + " = " + value);
                wrote = true;
            }
            in_sec = (to_lower(trim(t.substr(1, t.size() - 2))) == sec_l);
            if (in_sec) found_sec = true;
            out.push_back(line);
            continue;
        }
        if (in_sec) {
            auto eq = t.find('=');
            if (eq != std::string::npos) {
                auto k = to_lower(trim(t.substr(0, eq)));
                if (k == key_l) {
                    out.push_back(key + " = " + value);
                    wrote = true;
                    continue;
                }
            }
        }
        out.push_back(line);
    }
    if (in_sec && !wrote) {
        out.push_back(key + " = " + value);
        wrote = true;
    }
    if (!found_sec) {
        if (!out.empty() && !out.back().empty()) out.push_back("");
        out.push_back(sec_hdr);
        out.push_back(key + " = " + value);
    }

    std::ofstream o(ini_path, std::ios::trunc);
    for (size_t i = 0; i < out.size(); ++i) {
        o << out[i];
        if (i + 1 < out.size()) o << "\n";
    }
    if (!out.empty()) o << "\n";
}

bool ensure_wallet_configured(const std::filesystem::path& ini_path, bool interactive) {
    if (!std::filesystem::exists(ini_path)) {
        auto example = ini_path.parent_path() / "miner.ini.example";
        if (std::filesystem::exists(example)) {
            std::filesystem::copy_file(example, ini_path);
        } else {
            std::ofstream o(ini_path);
            o << "[account]\naddress =\nworker =\n";
        }
    }

    auto ini = parse_ini(ini_path);
    auto address = trim(get(ini, "account", "address"));
    auto worker = trim(get(ini, "account", "worker"));

    if (!valid_eth_address(address)) {
        if (!interactive) return false;
        std::cout << "\n=== " << "First-run wallet setup" << " ===\n";
        std::cout << "Enter your EVM wallet (0x + 40 hex chars):\n> " << std::flush;
        std::string input;
        if (!std::getline(std::cin, input)) return false;
        address = trim(input);
        if (!valid_eth_address(address)) {
            std::cerr << "Invalid wallet address.\n";
            return false;
        }
        set_ini_value(ini_path, "account", "address", address);
    }

    if (worker.empty()) {
        if (interactive) {
            std::cout << "Miner name (Enter for auto unique xnminer-xxxxxxxx):\n> " << std::flush;
            std::string input;
            std::getline(std::cin, input);
            worker = trim(input);
        }
        if (worker.empty()) worker = random_worker_name();
        set_ini_value(ini_path, "account", "worker", worker);
        auto woody = trim(get(ini, "monitoring", "woodyminer_custom_name"));
        if (woody.empty()) {
            set_ini_value(ini_path, "monitoring", "woodyminer_custom_name", worker);
        }
    }

    auto tid = trim(get(ini, "monitoring", "tracker_id"));
    if (tid.empty()) {
        tid = random_tracker_id();
        set_ini_value(ini_path, "monitoring", "tracker_id", tid);
    }
    return true;
}

Settings load_settings(const std::filesystem::path& ini_path) {
    Settings s;
    s.root = ini_path.parent_path();
    if (s.root.empty()) s.root = std::filesystem::current_path();

    auto ini = parse_ini(ini_path);

    s.address = trim(get(ini, "account", "address"));
    s.worker = trim(get(ini, "account", "worker"));
    s.base_url = trim(get(ini, "server", "base_url", s.base_url));
    s.connection_timeout_s = get_i(ini, "server", "connection_timeout_s", s.connection_timeout_s);
    s.network_poll_interval_s = get_i(ini, "server", "network_poll_interval_s", s.network_poll_interval_s);
    s.network_poll_timeout_s = get_i(ini, "server", "network_poll_timeout_s", 12);
    s.network_down_poll_interval_s =
        get_i(ini, "server", "network_down_poll_interval_s", s.network_down_poll_interval_s);
    s.lastblock_url = trim(get(ini, "server", "lastblock_url", s.lastblock_url));
    s.lastblock_url_fallback =
        trim(get(ini, "server", "lastblock_url_fallback", s.lastblock_url_fallback));
    s.lastblock_poll_interval_s =
        get_i(ini, "server", "lastblock_poll_interval_s", s.lastblock_poll_interval_s);
    s.lastblock_timeout_s = get_i(ini, "server", "lastblock_timeout_s", s.lastblock_timeout_s);

    s.strategy = to_lower(trim(get(ini, "mining", "strategy", s.strategy)));
    s.memory_cost = get_i(ini, "mining", "memory_cost", s.memory_cost);
    s.time_cost = get_i(ini, "mining", "time_cost", s.time_cost);
    s.parallelism = get_i(ini, "mining", "parallelism", s.parallelism);
    s.hash_len = get_i(ini, "mining", "hash_len", s.hash_len);
    s.force_mine_memory_cost =
        get_i(ini, "mining", "force_mine_memory_cost", s.force_mine_memory_cost);
    if (s.force_mine_memory_cost < 0) s.force_mine_memory_cost = 0;
    s.match_drain_enabled = get_b(ini, "mining", "match_drain_enabled", true);
    s.match_drain_min_queue = get_i(ini, "mining", "match_drain_min_queue", 1);
    s.match_drain_max_s = get_i(ini, "mining", "match_drain_max_s", 0);
    s.match_drain_parallel = get_i(ini, "mining", "match_drain_parallel", 0);
    s.match_drain_batch = get_i(ini, "mining", "match_drain_batch", 0);
    if (s.match_drain_min_queue < 1) s.match_drain_min_queue = 1;
    // 0 = drain until the bag is empty or both oracles leave.
    if (s.match_drain_max_s < 0) s.match_drain_max_s = 0;
    if (s.match_drain_max_s > 0 && s.match_drain_max_s < 5) s.match_drain_max_s = 5;
    if (s.match_drain_parallel < 0) s.match_drain_parallel = 0;
    if (s.match_drain_parallel > 256) s.match_drain_parallel = 256;
    s.xuni_mining_enabled = get_b(ini, "mining", "xuni_mining_enabled", true);
    s.xuni_queue_soft_cap = get_i(ini, "mining", "xuni_queue_soft_cap", s.xuni_queue_soft_cap);
    s.xuni_queue_resume = get_i(ini, "mining", "xuni_queue_resume", s.xuni_queue_resume);
    s.xuni_every_n_batches = get_i(ini, "mining", "xuni_every_n_batches", s.xuni_every_n_batches);
    s.xuni_max_lanes = get_i(ini, "mining", "xuni_max_lanes", s.xuni_max_lanes);
    if (s.xuni_queue_resume > s.xuni_queue_soft_cap) {
        s.xuni_queue_resume = std::max(0, s.xuni_queue_soft_cap / 2);
    }
    if (s.xuni_every_n_batches < 1) s.xuni_every_n_batches = 1;
    if (s.xuni_max_lanes < 0) s.xuni_max_lanes = 0;
    if (!s.xuni_mining_enabled) s.xuni_max_lanes = 0;

    s.target_vram_pct = get_d(ini, "efficiency", "target_vram_pct", s.target_vram_pct);
    s.desktop_headroom_pct = get_d(ini, "efficiency", "desktop_headroom_pct", s.desktop_headroom_pct);
    s.emergency_vram_pct = get_d(ini, "efficiency", "emergency_vram_pct", s.emergency_vram_pct);
    s.min_headroom_pct = get_d(ini, "efficiency", "min_headroom_pct", s.min_headroom_pct);
    s.runtime_overhead_pct = get_d(ini, "efficiency", "runtime_overhead_pct",
                                   get_d(ini, "cuda", "runtime_overhead_pct", s.runtime_overhead_pct));
    s.min_headroom_floor_mib = get_i(ini, "efficiency", "min_headroom_floor_mib", s.min_headroom_floor_mib);
    s.runtime_overhead_floor_mib =
        get_i(ini, "efficiency", "runtime_overhead_floor_mib", s.runtime_overhead_floor_mib);
    s.target_vram_mib = get_i(ini, "efficiency", "target_vram_mib", 0);
    s.headroom_mib = get_i(ini, "efficiency", "headroom_mib", 0);
    s.emergency_vram_mib = get_i(ini, "efficiency", "emergency_vram_mib", 0);
    s.min_headroom_mib = get_i(ini, "efficiency", "min_headroom_mib", 0);

    s.max_gpu_temp_c = get_i(ini, "efficiency", "max_gpu_temp_c", s.max_gpu_temp_c);
    s.warn_gpu_temp_c = get_i(ini, "efficiency", "warn_gpu_temp_c", s.warn_gpu_temp_c);
    s.max_mem_temp_c = get_i(ini, "efficiency", "max_mem_temp_c", s.max_mem_temp_c);
    s.warn_mem_temp_c = get_i(ini, "efficiency", "warn_mem_temp_c", s.warn_mem_temp_c);
    s.thermal_use_memory_junction =
        get_b(ini, "efficiency", "thermal_use_memory_junction", s.thermal_use_memory_junction);
    if (s.max_mem_temp_c < 50) s.max_mem_temp_c = 50;
    if (s.warn_mem_temp_c < 40) s.warn_mem_temp_c = 40;
    if (s.warn_mem_temp_c >= s.max_mem_temp_c) {
        s.warn_mem_temp_c = std::max(40, s.max_mem_temp_c - 4);
    }
    s.gpu_cooldown_s = get_i(ini, "efficiency", "gpu_cooldown_s", s.gpu_cooldown_s);
    s.gpu_power_boost_enabled = get_b(ini, "efficiency", "gpu_power_boost_enabled", true);
    s.gpu_power_target_pct = get_i(ini, "efficiency", "gpu_power_target_pct", s.gpu_power_target_pct);
    s.gpu_power_min_pct = get_i(ini, "efficiency", "gpu_power_min_pct", s.gpu_power_min_pct);
    s.gpu_difficulty_power_enabled = get_b(ini, "efficiency", "gpu_difficulty_power_enabled", true);
    s.gpu_difficulty_power_full_ratio =
        get_d(ini, "efficiency", "gpu_difficulty_power_full_ratio", s.gpu_difficulty_power_full_ratio);
    s.gpu_thermal_batch_enabled = get_b(ini, "efficiency", "gpu_thermal_batch_enabled", true);
    s.gpu_thermal_batch_min_scale =
        get_d(ini, "efficiency", "gpu_thermal_batch_min_scale", s.gpu_thermal_batch_min_scale);
    if (s.gpu_thermal_batch_min_scale < 0.20) s.gpu_thermal_batch_min_scale = 0.20;
    if (s.gpu_thermal_batch_min_scale > 1.0) s.gpu_thermal_batch_min_scale = 1.0;
    s.gpu_thermal_start_scale =
        get_d(ini, "efficiency", "gpu_thermal_start_scale", s.gpu_thermal_start_scale);
    if (s.gpu_thermal_start_scale < 0.50) s.gpu_thermal_start_scale = 0.50;
    if (s.gpu_thermal_start_scale > 1.0) s.gpu_thermal_start_scale = 1.0;
    s.thermal_batch_step = get_i(ini, "efficiency", "thermal_batch_step", s.thermal_batch_step);
    if (s.thermal_batch_step < 1) s.thermal_batch_step = 1;
    s.thermal_settle_s = get_i(ini, "efficiency", "thermal_settle_s", s.thermal_settle_s);
    if (s.thermal_settle_s < 5) s.thermal_settle_s = 5;
    if (s.thermal_settle_s > 180) s.thermal_settle_s = 180;
    s.gpu_windows_performance_mode = get_b(ini, "efficiency", "gpu_windows_performance_mode", false);
    s.sample_interval_s = get_i(ini, "efficiency", "sample_interval_s", s.sample_interval_s);

    s.db_path = resolve_path(s.root, get(ini, "queue", "db_path", "data/blocks.db"));
    s.jsonl_path = resolve_path(s.root, get(ini, "queue", "jsonl_path", "data/queue.jsonl"));
    s.rejected_jsonl_path =
        resolve_path(s.root, get(ini, "queue", "rejected_jsonl_path", "data/rejected.jsonl"));
    s.submit_cpu_fraction = get_d(ini, "queue", "submit_cpu_fraction", s.submit_cpu_fraction);
    s.bag_forward_url = trim(get(ini, "queue", "bag_forward_url"));
    s.bag_forward_token = trim(get(ini, "queue", "bag_forward_token"));
    s.bag_forward_batch = get_i(ini, "queue", "bag_forward_batch", s.bag_forward_batch);
    if (s.bag_forward_batch < 1) s.bag_forward_batch = 1;
    if (s.bag_forward_batch > 256) s.bag_forward_batch = 256;

    s.log_path = resolve_path(s.root, get(ini, "monitoring", "log_path", "data/session.log"));
    s.timelapse_path =
        resolve_path(s.root, get(ini, "monitoring", "timelapse_path", "data/session_timelapse.jsonl"));
    s.stats_interval_s = get_i(ini, "monitoring", "stats_interval_s", s.stats_interval_s);
    s.timelapse_sample_s = get_i(ini, "monitoring", "timelapse_sample_s", s.timelapse_sample_s);
    s.dashboard_enabled = get_b(ini, "monitoring", "dashboard_enabled", true);

    s.woodyminer_enabled = get_b(ini, "monitoring", "woodyminer_enabled", true);
    s.woodyminer_upload_url =
        trim(get(ini, "monitoring", "woodyminer_upload_url", s.woodyminer_upload_url));
    s.woodyminer_upload_period_s =
        get_i(ini, "monitoring", "woodyminer_upload_period_s", s.woodyminer_upload_period_s);
    s.woodyminer_custom_name = trim(get(ini, "monitoring", "woodyminer_custom_name"));
    if (s.woodyminer_custom_name.empty()) s.woodyminer_custom_name = s.worker;

    s.xenblockscan_enabled = get_b(ini, "monitoring", "xenblockscan_enabled", false);
    s.xenblockscan_endpoint =
        trim(get(ini, "monitoring", "xenblockscan_endpoint", s.xenblockscan_endpoint));
    s.xenblockscan_api_key = trim(get(ini, "monitoring", "xenblockscan_api_key"));
    s.xenblockscan_report_rejects = get_b(ini, "monitoring", "xenblockscan_report_rejects", false);
    s.xenblockscan_holdings_interval_s =
        get_i(ini, "monitoring", "xenblockscan_holdings_interval_s", s.xenblockscan_holdings_interval_s);
    s.xenblockscan_backfill = get_b(ini, "monitoring", "xenblockscan_backfill", false);
    s.tracker_id = trim(get(ini, "monitoring", "tracker_id"));

    s.update_check_enabled = get_b(ini, "update", "enabled", false);
    s.update_github_repo =
        trim(get(ini, "update", "github_repo", s.update_github_repo));
    s.update_github_ref = trim(get(ini, "update", "github_ref", s.update_github_ref));
    s.update_token = trim(get(ini, "update", "token"));
    s.update_check_interval_s = get_i(ini, "update", "check_interval_s", s.update_check_interval_s);
    if (s.update_check_interval_s < 60) s.update_check_interval_s = 60;
    s.update_sha_path =
        resolve_path(s.root, get(ini, "update", "sha_path", "data/build_sha"));

    s.device_id = get_i(ini, "cuda", "device_id", 0);
    s.cuda_batch_size = get_i(ini, "cuda", "batch_size", 0);
    s.cuda_max_batch_size = get_i(ini, "cuda", "max_batch_size", 0);
    s.cuda_runtime_overhead_mib = get_i(ini, "cuda", "runtime_overhead_mib", 0);
    s.vram_reference_difficulty =
        get_i(ini, "cuda", "vram_reference_difficulty", s.memory_cost);
    s.cuda_max_lanes = get_i(ini, "cuda", "max_lanes", 8);
    if (s.cuda_max_lanes < 1) s.cuda_max_lanes = 1;
    if (s.cuda_max_lanes > 16) s.cuda_max_lanes = 16;
    s.cuda_lane_reserve = get_i(ini, "cuda", "lane_reserve", 1);
    s.cpu_logical = cpu_logical_count();
    s.cpu_physical = cpu_physical_count();
    s.keygen_threads = get_i(ini, "cuda", "keygen_threads", 0);
    if (s.keygen_threads <= 0) {
        s.keygen_threads =
            auto_keygen_threads(s.cpu_logical, s.cpu_physical, s.cuda_max_lanes);
    } else {
        const int cap = std::max(2, s.cpu_logical > 2 ? s.cpu_logical - 1 : s.cpu_logical);
        s.keygen_threads = std::max(2, std::min(s.keygen_threads, cap));
    }
    if (s.match_drain_parallel <= 0) {
        s.match_drain_parallel = auto_match_drain_parallel(s.cpu_logical);
    } else {
        const int cap = std::max(8, std::min(256, s.cpu_logical * 8));
        s.match_drain_parallel = std::max(1, std::min(s.match_drain_parallel, cap));
    }
    s.work_patches = get_i(ini, "cuda", "work_patches", 2);
    if (s.work_patches < 2) s.work_patches = 2;
    if (s.work_patches > 3) s.work_patches = 3;

    return s;
}

}  // namespace xn
