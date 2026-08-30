#include "monitoring/local_stats.hpp"

#include "util/paths.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace xn {

LocalMiningStatsTracker::LocalMiningStatsTracker(std::filesystem::path path)
    : path_(std::move(path)) {
    ensure_parent_dir(path_);
    load();
}

std::string LocalMiningStatsTracker::today_key() const {
    // Mining day rolls at 01:00 local (same idea as Python DAY_ROLLOVER_HOUR=1).
    using clock = std::chrono::system_clock;
    auto t = clock::to_time_t(clock::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    if (tm.tm_hour < 1) {
        t -= 24 * 3600;
#ifdef _WIN32
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
    }
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d");
    return oss.str();
}

void LocalMiningStatsTracker::load() {
    std::ifstream in(path_);
    if (!in) return;
    try {
        auto j = nlohmann::json::parse(in, nullptr, false);
        if (j.is_discarded() || !j.is_object()) return;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!it.value().is_object()) continue;
            for (auto kit = it.value().begin(); kit != it.value().end(); ++kit) {
                data_[it.key()][kit.key()] = kit.value().get<int>();
            }
        }
    } catch (...) {
    }
}

void LocalMiningStatsTracker::save() const {
    nlohmann::json j = nlohmann::json::object();
    for (const auto& day : data_) {
        j[day.first] = day.second;
    }
    std::ofstream out(path_, std::ios::trunc);
    out << j.dump(2);
}

void LocalMiningStatsTracker::record_accept(const std::string& kind) {
    std::lock_guard<std::mutex> lock(mu_);
    data_[today_key()][kind]++;
    save();
}

std::unordered_map<std::string, int> LocalMiningStatsTracker::today_counts() const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = data_.find(today_key());
    if (it == data_.end()) return {{"XUNI", 0}, {"XNM", 0}, {"XBLK", 0}};
    std::unordered_map<std::string, int> out{{"XUNI", 0}, {"XNM", 0}, {"XBLK", 0}};
    for (const auto& kv : it->second) out[kv.first] = kv.second;
    return out;
}

}  // namespace xn
