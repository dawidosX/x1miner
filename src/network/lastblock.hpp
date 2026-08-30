#pragma once

#include "common.hpp"

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace xn {

struct PaperStatus {
    bool ok = false;
    std::optional<int> newest_m;
    std::optional<int> majority_m;
    int tip_id = 0;
    bool mixed = false;
    double latency_ms = -1.0;
    std::string error;
    uint64_t seq = 0;
};

class LastblockPoller {
public:
    LastblockPoller(std::string url, std::string fallback_url, int poll_interval_s, int timeout_s);
    ~LastblockPoller();

    void start();
    void stop();
    PaperStatus get_status() const;
    PaperStatus poll_once(int timeout_s = -1);

private:
    void loop();
    PaperStatus fetch(const std::string& url, int timeout_s);

    std::string url_;
    std::string fallback_url_;
    int poll_interval_s_ = 2;
    int timeout_s_ = 8;
    std::atomic<bool> running_{false};
    mutable std::mutex mu_;
    PaperStatus status_;
    uint64_t seq_ = 0;
    std::thread thread_;
};

}  // namespace xn
