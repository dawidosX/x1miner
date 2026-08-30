#include "network/updater.hpp"

#include "util/http.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace xn {
namespace {

std::string trim_copy(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

bool looks_like_sha(const std::string& s) {
    if (s.size() < 7 || s.size() > 40) return false;
    for (char c : s) {
        if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

std::string parse_sha(const std::string& body) {
    auto t = trim_copy(body);
    if (!t.empty() && t.back() == '\n') t.pop_back();
    if (looks_like_sha(t) && t.find('{') == std::string::npos) return t;
    const std::string keys[] = {"\"sha\":\"", "\"sha\": \""};
    for (const auto& key : keys) {
        auto p = body.find(key);
        if (p == std::string::npos) continue;
        p += key.size();
        auto end = body.find('"', p);
        if (end == std::string::npos || end <= p) continue;
        auto sha = body.substr(p, end - p);
        if (looks_like_sha(sha)) return sha;
    }
    return {};
}

}  // namespace

std::string read_build_sha(const std::string& sha_path) {
    if (sha_path.empty()) return {};
    std::ifstream in(sha_path);
    if (!in) return {};
    std::string line;
    std::getline(in, line);
    return trim_copy(line);
}

std::string github_token_from_env() {
    const char* keys[] = {"GH_TOKEN", "GITHUB_TOKEN", "GIT_TOKEN"};
    for (const char* k : keys) {
        if (const char* v = std::getenv(k)) {
            auto s = trim_copy(v);
            if (!s.empty() && s != "YOUR_PAT") return s;
        }
    }
    return {};
}

UpdateCheckResult check_github_update(const std::string& repo, const std::string& ref,
                                      const std::string& token, const std::string& local_sha) {
    UpdateCheckResult out;
    out.local_sha = local_sha;
    if (repo.empty() || local_sha.empty()) {
        out.error = "no repo or local sha";
        return out;
    }
    const std::string branch = ref.empty() ? "main" : ref;
    const std::string url = "https://api.github.com/repos/" + repo + "/commits/" + branch;
    std::string hdr;
    if (!token.empty()) hdr = "Authorization: Bearer " + token;
    auto resp = http_get(url, 8000, hdr);
    if (resp.status != 200) {
        out.error = resp.error.empty() ? ("HTTP " + std::to_string(resp.status)) : resp.error;
        return out;
    }
    out.remote_sha = parse_sha(resp.body);
    if (out.remote_sha.empty()) {
        out.error = "no sha in GitHub response";
        return out;
    }
    auto loc = out.local_sha;
    auto rem = out.remote_sha;
    if (loc.size() > rem.size()) loc.resize(rem.size());
    if (rem.size() > loc.size()) rem.resize(loc.size());
    for (char& c : loc) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (char& c : rem) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    out.available = (loc != rem);
    return out;
}

}  // namespace xn
