#pragma once

#include "common.hpp"

#include <mutex>
#include <string>

namespace xn {

class MetricsTracker {
public:
    void record_hashes(int64_t n, double hps_sample);
    void record_found(const std::string& kind);
    void record_enqueued(const std::string& kind);
    void record_accepted_live(const std::string& kind);
    void record_accepted_flush(const std::string& kind);
    void record_rejected_live(const std::string& kind);
    void record_rejected_flush(const std::string& kind);
    void record_resubmission(const std::string& kind);
    void sync_pending(int pending);
    void set_active_lanes(int lanes);
    MiningStats stats() const;

private:
    mutable std::mutex mu_;
    MiningStats s_;
};

}  // namespace xn
