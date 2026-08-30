#pragma once

#include "common.hpp"

#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace xn {

int accept_network_difficulty(int raw, int fallback);

class NetworkPoller {
public:
    NetworkPoller(std::string difficulty_url, int poll_interval_s, int down_poll_interval_s,
                  int timeout_s);
    ~NetworkPoller();

    void start();
    void stop();
    NetworkStatus get_status() const;
    NetworkStatus poll_once(int timeout_s = -1);

private:
    void loop();

    std::string url_;
    int poll_interval_s_ = 15;
    int down_poll_interval_s_ = 30;
    int timeout_s_ = 3;
    std::atomic<bool> running_{false};
    mutable std::mutex mu_;
    NetworkStatus status_;
    uint64_t seq_ = 0;
    std::thread thread_;
};

}  // namespace xn
