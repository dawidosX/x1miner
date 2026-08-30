#include "util/paths.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace xn {

std::filesystem::path resolve_path(const std::filesystem::path& root, const std::string& rel) {
    std::filesystem::path p(rel);
    if (p.is_absolute()) return p;
    return root / p;
}

void ensure_parent_dir(const std::filesystem::path& path) {
    auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

std::string now_iso_local() {
    using clock = std::chrono::system_clock;
    auto t = clock::to_time_t(clock::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

std::string now_iso_utc() {
    using clock = std::chrono::system_clock;
    auto t = clock::to_time_t(clock::now());
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

}  // namespace xn
