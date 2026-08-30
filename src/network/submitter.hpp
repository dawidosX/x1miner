#pragma once

#include "common.hpp"
#include "util/logger.hpp"

#include <string>

namespace xn {

bool submit_accepted(int status, const std::string& body);
bool is_difficulty_mismatch(int status, const std::string& body);
bool is_xuni_window_reject(int status, const std::string& body);
bool is_transient_submit_failure(int status, const std::string& body);
bool is_pool_hold(int status, const std::string& body);
bool counts_as_reject(int status, const std::string& body);
std::string submit_response_hint(int status, const std::string& body);

class Submitter {
public:
    Submitter(std::string verify_url, std::string account, std::string worker,
              SessionLogger* logger = nullptr);

    SubmitResult submit(const BlockHit& hit, int timeout_s = 20);

private:
    std::string verify_url_;
    std::string account_;
    std::string worker_;
    SessionLogger* logger_ = nullptr;
};

}  // namespace xn
