#pragma once

#include "common.hpp"

#include <filesystem>
#include <string>

namespace xn {

class SessionTimelapse {
public:
    SessionTimelapse(std::filesystem::path path, int sample_s);
    void maybe_sample(const MiningStats& stats, const GpuSnapshot* gpu, int pending, bool network_ok);
    void record_event(const std::string& label);
    void finalize();

private:
    std::filesystem::path path_;
    int sample_s_ = 30;
    double last_sample_ = 0;
};

}  // namespace xn
