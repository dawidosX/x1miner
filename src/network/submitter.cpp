#include "network/submitter.hpp"

#include "util/http.hpp"
#include "util/paths.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <sstream>

namespace xn {
namespace {

std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

}  // namespace

bool submit_accepted(int status, const std::string& body) {
    if (status >= 200 && status < 300) return true;
    auto l = lower(body);
    return l.find("already exists") != std::string::npos;
}

bool is_difficulty_mismatch(int status, const std::string& body) {
    if (status == 0) return false;
    auto l = lower(body);
    if (l.find("hash does not contain 'm=") != std::string::npos) return true;
    if (l.find("does not contain") != std::string::npos && l.find("m=") != std::string::npos)
        return true;
    if (l.find("memory_cost") != std::string::npos && l.find("does not contain") != std::string::npos)
        return true;
    // Pool uses HTTP 401 + this text for "won't take this hash now" (wrong live m=,
    // or a slammed /verify). It is not a broken digest — keep the bag and retry.
    if (l.find("hash verification failed") != std::string::npos) return true;
    return false;
}

bool is_xuni_window_reject(int status, const std::string& body) {
    if (status == 0) return false;
    auto l = lower(body);
    return l.find("outside of time window") != std::string::npos ||
           l.find("outside of proper time frame") != std::string::npos ||
           l.find("time frame") != std::string::npos || l.find("time window") != std::string::npos;
}

bool is_transient_submit_failure(int status, const std::string& body) {
    if (status == 0) return true;
    auto l = lower(body);
    return l.find("timed out") != std::string::npos || l.find("timeout") != std::string::npos ||
           l.find("connection") != std::string::npos;
}

bool is_pool_hold(int status, const std::string& body) {
    if (status == 401 || status == 403 || status == 429) return true;
    auto l = lower(body);
    return l.find("hash verification failed") != std::string::npos;
}

bool counts_as_reject(int status, const std::string& body) {
    if (submit_accepted(status, body)) return false;
    if (is_difficulty_mismatch(status, body)) return false;
    if (is_xuni_window_reject(status, body)) return false;
    if (is_transient_submit_failure(status, body)) return false;
    if (is_pool_hold(status, body)) return false;
    return true;
}

std::string submit_response_hint(int status, const std::string& body) {
    if (status >= 200 && status < 300) return "HTTP " + std::to_string(status);
    auto l = lower(body);
    if (l.find("already exists") != std::string::npos) return "already on server (duplicate)";
    if (status == 401 && body.empty()) return "HTTP 401 empty body";
    if (is_pool_hold(status, body)) return "pool hold — retry (hash kept)";
    if (is_difficulty_mismatch(status, body)) return "difficulty mismatch — hold for matching m=";
    if (is_xuni_window_reject(status, body)) return "outside XUNI window";
    if (is_transient_submit_failure(status, body)) return "network error";
    std::string snippet = body;
    if (snippet.size() > 80) snippet.resize(80);
    return "HTTP " + std::to_string(status) + (snippet.empty() ? "" : " — " + snippet);
}

Submitter::Submitter(std::string verify_url, std::string account, std::string worker,
                     SessionLogger* logger)
    : verify_url_(std::move(verify_url)),
      account_(std::move(account)),
      worker_(std::move(worker)),
      logger_(logger) {}

SubmitResult Submitter::submit(const BlockHit& hit, int timeout_s) {
    // Exact field types that landed HTTP 200s on 2026-08-12 (PowerShell flusher).
    nlohmann::json payload = {{"account", account_},
                              {"key", hit.key},
                              {"hash_to_verify", hit.hash_str},
                              {"attempts", std::to_string(hit.attempts)},
                              {"hashes_per_second", std::to_string(hit.hps)},
                              {"worker", worker_}};

    SubmitResult result;
    result.submitted_at = now_iso_local();
    // python-requests UA + charset JSON is the path that accepted blocks.
    // Miner UA xnminer-cuda/4.0 was 401 empty-body on every m=100 window today.
    auto resp = http_post_json(verify_url_, payload.dump(), timeout_s * 1000,
                               "python-requests/2.31.0");
    result.status = resp.status;
    result.body = resp.body.empty() ? resp.error : resp.body;
    result.ok = submit_accepted(result.status, result.body);

    if (logger_ && logger_->echo_console) {
        if (result.ok) {
            logger_->info("SUBMIT " + hit.block_type + " status=" + std::to_string(result.status) +
                          " ok=1");
        } else {
            logger_->warn("SUBMIT " + hit.block_type + " status=" + std::to_string(result.status) +
                          " ok=0");
        }
    }
    return result;
}

}  // namespace xn
