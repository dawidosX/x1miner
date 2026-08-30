#include "network/lastblock.hpp"

#include "mining/argon2_encode.hpp"
#include "util/http.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <map>
#include <thread>

namespace xn {

LastblockPoller::LastblockPoller(std::string url, std::string fallback_url, int poll_interval_s,
                                 int timeout_s)
    : url_(std::move(url)),
      fallback_url_(std::move(fallback_url)),
      poll_interval_s_(poll_interval_s > 0 ? poll_interval_s : 2),
      timeout_s_(timeout_s > 0 ? timeout_s : 8) {}

LastblockPoller::~LastblockPoller() { stop(); }

void LastblockPoller::start() {
    if (running_) return;
    running_ = true;
    thread_ = std::thread([this] { loop(); });
}

void LastblockPoller::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

PaperStatus LastblockPoller::get_status() const {
    std::lock_guard<std::mutex> lock(mu_);
    return status_;
}

PaperStatus LastblockPoller::fetch(const std::string& url, int timeout_s) {
    auto t0 = std::chrono::steady_clock::now();
    auto resp = http_get(url, timeout_s * 1000);
    auto t1 = std::chrono::steady_clock::now();
    PaperStatus st;
    st.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (resp.status == 0 || !resp.error.empty()) {
        st.error = resp.error.empty() ? "no response" : resp.error;
        return st;
    }
    if (resp.status < 200 || resp.status >= 300) {
        st.error = "HTTP " + std::to_string(resp.status);
        return st;
    }
    try {
        auto j = nlohmann::json::parse(resp.body);
        if (!j.is_array() || j.empty()) {
            st.error = "empty lastblock";
            return st;
        }
        int best_id = -1;
        std::optional<int> best_m;
        std::map<int, int> counts;
        for (const auto& row : j) {
            if (!row.is_object()) continue;
            int id = 0;
            try {
                id = row.value("block_id", 0);
            } catch (...) {
                continue;
            }
            std::string h;
            try {
                h = row.value("hash_to_verify", std::string{});
            } catch (...) {
            }
            auto m = memory_cost_from_hash(h);
            if (m) counts[*m] += 1;
            if (id > best_id) {
                best_id = id;
                best_m = m;
            }
        }
        if (best_id <= 0 || !best_m) {
            st.error = "no m= in lastblock";
            return st;
        }
        st.ok = true;
        st.tip_id = best_id;
        st.newest_m = best_m;
        int maj = 0, maj_n = 0;
        for (const auto& kv : counts) {
            if (kv.second > maj_n) {
                maj_n = kv.second;
                maj = kv.first;
            }
        }
        if (maj > 0) st.majority_m = maj;
        st.mixed = counts.size() > 1;
    } catch (const std::exception& ex) {
        st.error = ex.what();
    }
    return st;
}

PaperStatus LastblockPoller::poll_once(int timeout_s) {
    int t = timeout_s > 0 ? timeout_s : timeout_s_;
    PaperStatus st = fetch(url_, t);
    if (!st.ok && !fallback_url_.empty() && fallback_url_ != url_) {
        auto fb = fetch(fallback_url_, t);
        if (fb.ok) st = std::move(fb);
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        st.seq = ++seq_;
        status_ = st;
    }
    return st;
}

void LastblockPoller::loop() {
    while (running_) {
        poll_once();
        int sleep_s = poll_interval_s_;
        for (int i = 0; i < sleep_s * 10 && running_; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

}  // namespace xn
