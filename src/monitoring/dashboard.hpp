#pragma once

#include "common.hpp"
#include "config/settings.hpp"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xn {

class MinerDashboard {
public:
    explicit MinerDashboard(const Settings& settings);

    void start();
    void stop();
    void set_status(const std::string& status);
    void set_network(bool ok, std::optional<int> difficulty, bool stale = false);
    void set_mining_m(int mining_m, bool force_hybrid);
    void set_cuda_batch(int batch, int lanes, double thermal_scale = 1.0);
    void set_wallet_line(const std::string& line);
    void set_uptime_s(int uptime_s);
    void event(const std::string& action, const std::string& block, const std::string& detail = "");
    void update(const MiningStats& stats, const GpuSnapshot* gpu,
                const std::unordered_map<std::string, int>& pending_by_type,
                const std::unordered_map<std::string, int>& resubmission_by_type);
    void render();

private:
    void maybe_export_status_json();

    Settings settings_;
    std::mutex mu_;
    MiningStats stats_;
    std::optional<GpuSnapshot> gpu_;
    std::string status_ = "Starting...";
    bool network_ok_ = false;
    bool network_stale_ = false;
    std::optional<int> difficulty_;
    int mining_m_ = 0;
    bool force_hybrid_ = false;
    int cuda_batch_ = 0;
    int cuda_lanes_ = 1;
    double thermal_scale_ = 1.0;
    int uptime_s_ = 0;
    int pending_xuni_ = 0;
    int pending_xnm_ = 0;
    int pending_xblk_ = 0;
    std::string wallet_line_;
    std::vector<std::string> events_;
    double last_json_export_at_ = 0.0;
    bool active_ = false;
};

}  // namespace xn
