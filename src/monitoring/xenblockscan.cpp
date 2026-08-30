#include "monitoring/xenblockscan.hpp"

#include "common.hpp"
#include "util/http.hpp"
#include "util/paths.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <thread>

namespace xn {
namespace {

std::string events_url(std::string endpoint) {
    while (!endpoint.empty() && endpoint.back() == '/') endpoint.pop_back();
    if (endpoint.size() >= 9 && endpoint.substr(endpoint.size() - 9) == "/holdings") {
        return endpoint.substr(0, endpoint.size() - 9) + "/events";
    }
    if (endpoint.size() >= 7 && endpoint.substr(endpoint.size() - 7) == "/events") return endpoint;
    if (endpoint.size() >= 7 && endpoint.substr(endpoint.size() - 7) == "/api/v1")
        return endpoint + "/events";
    return endpoint + "/api/v1/events";
}

std::string holdings_url(const std::string& endpoint) {
    auto e = events_url(endpoint);
    if (e.size() >= 7 && e.substr(e.size() - 7) == "/events") {
        return e.substr(0, e.size() - 7) + "/holdings";
    }
    return e + "/holdings";
}

}  // namespace

void XenblockscanReporter::configure(bool enabled, std::string endpoint, std::string api_key,
                                     bool report_rejects) {
    enabled_ = enabled;
    endpoint_ = std::move(endpoint);
    api_key_ = std::move(api_key);
    report_rejects_ = report_rejects;
    if (enabled_) ensure_worker();
}

void XenblockscanReporter::ensure_worker() {
    if (running_) return;
    running_ = true;
    thread_ = std::thread([this] { loop(); });
}

void XenblockscanReporter::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void XenblockscanReporter::enqueue(std::string url, std::string body) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mu_);
    if (q_.size() > 256) q_.pop();
    q_.emplace(std::move(url), std::move(body));
}

void XenblockscanReporter::loop() {
    while (running_) {
        std::pair<std::string, std::string> job;
        bool has = false;
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!q_.empty()) {
                job = std::move(q_.front());
                q_.pop();
                has = true;
            }
        }
        if (has) {
            // BUGFIX: the API key was read from config but never attached to
            // the request — every authed endpoint returned 401. (Linux path
            // honors extra_header; the module is Linux-only in practice.)
            const std::string auth =
                api_key_.empty() ? std::string{} : ("X-Api-Key: " + api_key_);
            http_post_json(job.first, job.second, 1500, "xnminer-cuda/4.0", auth);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

void XenblockscanReporter::report_accepted(const std::string& account, const std::string& kind,
                                           const std::string& key, const std::string& hash,
                                           const std::string& worker, int difficulty) {
    nlohmann::json body = {{"type", "accept"},
                           {"account", account},
                           {"kind", kind},
                           {"key", key},
                           {"hash_to_verify", hash},
                           {"worker", worker},
                           {"difficulty", difficulty},
                           {"plugin", "tony-xnminer-cuda"},
                           {"version", kMinerVersion},
                           {"occurred_at", now_iso_utc()}};
    enqueue(events_url(endpoint_), body.dump());
}

void XenblockscanReporter::report_holdings(const std::string& account, const std::string& worker,
                                           std::optional<double> xnm, std::optional<double> xuni,
                                           std::optional<double> xblk, std::optional<double> hashrate,
                                           const std::string& tracker_id) {
    nlohmann::json body = {{"account", account}, {"worker", worker}, {"tracker_id", tracker_id}};
    if (xnm) body["xnm"] = *xnm;
    if (xuni) body["xuni"] = *xuni;
    if (xblk) body["xblk"] = *xblk;
    if (hashrate) body["hashrate"] = *hashrate;
    enqueue(holdings_url(endpoint_), body.dump());
}

void XenblockscanReporter::report_tracker(const std::string& tracker_id, const std::string& account,
                                          const std::string& worker, std::optional<double> hashrate,
                                          int accepted, int rejected, int found,
                                          std::optional<int> difficulty, bool network_ok) {
    nlohmann::json body = {{"type", "tracker"},
                           {"tracker_id", tracker_id},
                           {"account", account},
                           {"worker", worker},
                           {"accepted", accepted},
                           {"rejected", rejected},
                           {"found", found},
                           {"network_ok", network_ok}};
    if (hashrate) body["hashrate"] = *hashrate;
    if (difficulty) body["difficulty"] = *difficulty;
    enqueue(events_url(endpoint_), body.dump());
}

}  // namespace xn
