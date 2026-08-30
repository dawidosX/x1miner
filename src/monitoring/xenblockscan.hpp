#pragma once

#include <atomic>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>

namespace xn {

class XenblockscanReporter {
public:
    void configure(bool enabled, std::string endpoint, std::string api_key, bool report_rejects);
    void stop();

    void report_accepted(const std::string& account, const std::string& kind, const std::string& key,
                         const std::string& hash, const std::string& worker, int difficulty);
    void report_holdings(const std::string& account, const std::string& worker,
                         std::optional<double> xnm, std::optional<double> xuni,
                         std::optional<double> xblk, std::optional<double> hashrate,
                         const std::string& tracker_id);
    void report_tracker(const std::string& tracker_id, const std::string& account,
                        const std::string& worker, std::optional<double> hashrate, int accepted,
                        int rejected, int found, std::optional<int> difficulty, bool network_ok);

private:
    void ensure_worker();
    void loop();
    void enqueue(std::string url, std::string body);

    bool enabled_ = false;
    std::string endpoint_;
    std::string api_key_;
    bool report_rejects_ = false;
    std::mutex mu_;
    std::queue<std::pair<std::string, std::string>> q_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

}  // namespace xn
