#pragma once

#include <string>

namespace xn {

struct UpdateCheckResult {
    bool available = false;
    std::string local_sha;
    std::string remote_sha;
    std::string error;
};

std::string read_build_sha(const std::string& sha_path);
std::string github_token_from_env();
UpdateCheckResult check_github_update(const std::string& repo, const std::string& ref,
                                      const std::string& token, const std::string& local_sha);

}  // namespace xn
