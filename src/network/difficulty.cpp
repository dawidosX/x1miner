#include "network/difficulty.hpp"

#include "util/http.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cctype>
#include <optional>

namespace xn {
namespace {

std::optional<int> parse_difficulty_body(std::string body) {
    while (!body.empty() && std::isspace(static_cast<unsigned char>(body.front()))) {
        body.erase(body.begin());
    }
    while (!body.empty() && std::isspace(static_cast<unsigned char>(body.back()))) {
        body.pop_back();
    }
    if (body.empty()) return std::nullopt;

    // Plain integer: "1100"
    try {
        size_t idx = 0;
        int d = std::stoi(body, &idx);
        if (idx > 0) return d;
    } catch (...) {
    }

    // JSON forms: {"difficulty":"1100"} or {"difficulty":1100}
    try {
        auto j = nlohmann::json::parse(body);
        if (j.contains("difficulty")) {
            const auto& v = j["difficulty"];
            if (v.is_number_integer()) return v.get<int>();
            if (v.is_string()) return std::stoi(v.get<std::string>());
        }
        if (j.is_number_integer()) return j.get<int>();
        if (j.is_string()) return std::stoi(j.get<std::string>());
    } catch (...) {
    }
    return std::nullopt;
}

}  // namespace

int accept_network_difficulty(int raw, int fallback) {
    if (raw <= 0) return fallback > 0 ? fallback : 1100;
    // Sanity: Argon2 m= is typically tens to tens of thousands.
    if (raw > 10'000'000) return fallback > 0 ? fallback : 1100;
    return raw;
}

NetworkPoller::NetworkPoller(std::string difficulty_url, int poll_interval_s,
                             int down_poll_interval_s, int timeout_s)
    : url_(std::move(difficulty_url)),
      poll_interval_s_(poll_interval_s),
      down_poll_interval_s_(down_poll_interval_s),
      timeout_s_(timeout_s) {}

NetworkPoller::~NetworkPoller() { stop(); }

void NetworkPoller::start() {
    if (running_) return;
    running_ = true;
    thread_ = std::thread([this] { loop(); });
}

void NetworkPoller::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

NetworkStatus NetworkPoller::get_status() const {
    std::lock_guard<std::mutex> lock(mu_);
    return status_;
}

NetworkStatus NetworkPoller::poll_once(int timeout_s) {
    int t = timeout_s > 0 ? timeout_s : timeout_s_;
    auto t0 = std::chrono::steady_clock::now();
    auto resp = http_get(url_, t * 1000);
    auto t1 = std::chrono::steady_clock::now();
    NetworkStatus st;
    st.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (resp.status >= 200 && resp.status < 300) {
        auto d = parse_difficulty_body(resp.body);
        if (d) {
            st.difficulty = *d;
            st.ok = true;
        } else {
            st.ok = false;
            st.error = "invalid difficulty body";
        }
    } else {
        st.ok = false;
        st.error = resp.error.empty() ? ("HTTP " + std::to_string(resp.status)) : resp.error;
        if (st.error.empty()) st.error = "no response";
    }

    // :80 /difficulty often times out. HTTPS leaderboard is reachable, but its
    // "difficulty":"100" is a documented stub (xenblocks-api-findings-log §12.5).
    // Never start an m=100 flush from that stub or from a stale lastblock bag.
    if (!st.ok) {
        auto board = http_get("https://xenblocks.io/v1/leaderboard?limit=1", t * 1000);
        if (board.status >= 200 && board.status < 300) {
            try {
                auto j = nlohmann::json::parse(board.body, nullptr, false);
                if (j.contains("difficulty")) {
                    int d = 0;
                    const auto& v = j["difficulty"];
                    if (v.is_number_integer()) d = v.get<int>();
                    else if (v.is_string()) d = std::stoi(v.get<std::string>());
                    if (d > 0 && d != 100) {
                        st.difficulty = d;
                        st.ok = true;
                        st.error.clear();
                    }
                }
            } catch (...) {
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        st.seq = ++seq_;
        status_ = st;
    }
    return st;
}

void NetworkPoller::loop() {
    while (running_) {
        auto st = poll_once();
        int sleep_s = st.ok ? poll_interval_s_ : down_poll_interval_s_;
        for (int i = 0; i < sleep_s * 10 && running_; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

}  // namespace xn
