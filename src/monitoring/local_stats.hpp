#pragma once

#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

namespace xn {

class LocalMiningStatsTracker {
public:
    explicit LocalMiningStatsTracker(std::filesystem::path path);
    void record_accept(const std::string& kind);
    std::unordered_map<std::string, int> today_counts() const;

private:
    void load();
    void save() const;
    std::string today_key() const;

    std::filesystem::path path_;
    mutable std::mutex mu_;
    // day -> kind -> count
    std::unordered_map<std::string, std::unordered_map<std::string, int>> data_;
};

}  // namespace xn
