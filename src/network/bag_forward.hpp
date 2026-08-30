#pragma once

#include "queue/store.hpp"
#include "util/logger.hpp"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace xn {

// Ship queued hits to the Windows home vault (POST /bag). Local bag stays for flush.
class BagForwarder {
public:
    BagForwarder(BlockStore& store, std::string url, std::string token, std::string worker,
                 int batch, SessionLogger* logger);
    ~BagForwarder();

    void start();
    void stop();
    void notify();

private:
    void loop();
    int send_wave();

    BlockStore& store_;
    std::string url_;
    std::string token_;
    std::string worker_;
    int batch_ = 32;
    SessionLogger* logger_ = nullptr;
    std::atomic<bool> running_{false};
    std::mutex mu_;
    std::condition_variable cv_;
    std::thread thread_;
    int fail_streak_ = 0;
};

}  // namespace xn
