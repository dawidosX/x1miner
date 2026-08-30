#include "monitoring/metrics.hpp"

namespace xn {

void MetricsTracker::record_hashes(int64_t n, double hps_sample) {
    std::lock_guard<std::mutex> lock(mu_);
    s_.total_hashes += n;
    if (hps_sample > 0) {
        if (s_.hps_ema <= 0) s_.hps_ema = hps_sample;
        else s_.hps_ema = s_.hps_ema * 0.7 + hps_sample * 0.3;
    }
}

void MetricsTracker::record_found(const std::string& kind) {
    std::lock_guard<std::mutex> lock(mu_);
    stat_field_for_kind(s_, kind, "found")++;
    s_.session_hits++;
}

void MetricsTracker::record_enqueued(const std::string& kind) {
    std::lock_guard<std::mutex> lock(mu_);
    stat_field_for_kind(s_, kind, "enqueued")++;
}

void MetricsTracker::record_accepted_live(const std::string& kind) {
    std::lock_guard<std::mutex> lock(mu_);
    stat_field_for_kind(s_, kind, "accepted_live")++;
}

void MetricsTracker::record_accepted_flush(const std::string& kind) {
    std::lock_guard<std::mutex> lock(mu_);
    stat_field_for_kind(s_, kind, "accepted_flush")++;
}

void MetricsTracker::record_rejected_live(const std::string& kind) {
    std::lock_guard<std::mutex> lock(mu_);
    stat_field_for_kind(s_, kind, "rejected_live")++;
}

void MetricsTracker::record_rejected_flush(const std::string& kind) {
    std::lock_guard<std::mutex> lock(mu_);
    stat_field_for_kind(s_, kind, "rejected_flush")++;
}

void MetricsTracker::record_resubmission(const std::string& kind) {
    std::lock_guard<std::mutex> lock(mu_);
    stat_field_for_kind(s_, kind, "resubmission")++;
}

void MetricsTracker::sync_pending(int pending) {
    std::lock_guard<std::mutex> lock(mu_);
    s_.queued = pending;
}

void MetricsTracker::set_active_lanes(int lanes) {
    std::lock_guard<std::mutex> lock(mu_);
    s_.active_lanes = lanes;
}

MiningStats MetricsTracker::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return s_;
}

}  // namespace xn
