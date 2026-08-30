#include "network/bag_forward.hpp"

#include "util/http.hpp"

#include <chrono>
#include <vector>

namespace xn {

BagForwarder::BagForwarder(BlockStore& store, std::string url, std::string token,
                           std::string worker, int batch, SessionLogger* logger)
    : store_(store),
      url_(std::move(url)),
      token_(std::move(token)),
      worker_(std::move(worker)),
      batch_(batch > 0 ? batch : 32),
      logger_(logger) {}

BagForwarder::~BagForwarder() { stop(); }

void BagForwarder::start() {
    if (running_ || url_.empty()) return;
    running_ = true;
    thread_ = std::thread([this] { loop(); });
    if (logger_) logger_->info("Bag forward to " + url_ + " (keep local flush, Windows is vault)");
}

void BagForwarder::stop() {
    if (!running_ && !thread_.joinable()) return;
    running_ = false;
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void BagForwarder::notify() { cv_.notify_all(); }

void BagForwarder::loop() {
    // Catch up anything already on disk, then ship new enqueues quickly.
    while (running_) {
        const int sent = send_wave();
        std::unique_lock<std::mutex> lock(mu_);
        if (!running_) break;
        const int wait_ms = sent > 0 ? 50 : (fail_streak_ > 0 ? 5000 : 1000);
        cv_.wait_for(lock, std::chrono::milliseconds(wait_ms), [this] { return !running_.load(); });
    }
    if (!url_.empty()) send_wave();
}

int BagForwarder::send_wave() {
    if (url_.empty()) return 0;
    auto pending = store_.list_unforwarded(batch_);
    if (pending.empty()) {
        fail_streak_ = 0;
        return 0;
    }
    std::vector<BlockHit> hits;
    std::vector<int64_t> ids;
    hits.reserve(pending.size());
    ids.reserve(pending.size());
    for (const auto& pb : pending) {
        hits.push_back(pb.hit);
        ids.push_back(pb.id);
    }
    const std::string body = hits_to_bag_json(hits, worker_);
    const std::string auth = token_.empty() ? std::string() : ("Authorization: Bearer " + token_);
    auto resp = http_post_json(url_, body, 8000, "xnminer-cuda/bag-forward", auth);
    if (resp.status == 200 && (resp.body.empty() || resp.body.find("\"ok\":false") == std::string::npos)) {
        store_.mark_forwarded_many(ids);
        fail_streak_ = 0;
        if (logger_) {
            logger_->info("Bag forwarded " + std::to_string(ids.size()) + " hit(s) to Windows vault");
        }
        return static_cast<int>(ids.size());
    }
    ++fail_streak_;
    if (logger_ && (fail_streak_ == 1 || fail_streak_ % 12 == 0)) {
        logger_->warn("Bag forward failed" +
                      (resp.status ? (" HTTP " + std::to_string(resp.status)) : std::string()) +
                      (resp.error.empty() ? "" : (" " + resp.error)) +
                      " — keeping local bag, retrying");
    }
    return 0;
}

}  // namespace xn
