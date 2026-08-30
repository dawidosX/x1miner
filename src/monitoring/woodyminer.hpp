#pragma once

#include "common.hpp"
#include "util/logger.hpp"

#include <atomic>
#include <functional>
#include <string>
#include <thread>

namespace xn {

std::string derive_machine_id(int device_index = 0);

class WoodyminerUploader {
public:
    WoodyminerUploader(std::string upload_url, int period_s, std::string custom_name,
                       std::string miner_address, std::string machine_id,
                       std::function<MiningStats()> get_stats,
                       std::function<std::optional<GpuSnapshot>()> get_gpu,
                       std::function<int()> get_difficulty, double session_started_at,
                       SessionLogger* logger);

    void start();
    void stop();

private:
    void loop();

    std::string upload_url_;
    int period_s_ = 60;
    std::string custom_name_;
    std::string miner_address_;
    std::string machine_id_;
    std::function<MiningStats()> get_stats_;
    std::function<std::optional<GpuSnapshot>()> get_gpu_;
    std::function<int()> get_difficulty_;
    double session_started_at_ = 0;
    SessionLogger* logger_ = nullptr;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

}  // namespace xn
