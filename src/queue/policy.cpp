#include "queue/policy.hpp"

#include <cctype>
#include <chrono>
#include <ctime>

namespace xn {

int local_minute_of_hour() {
    using clock = std::chrono::system_clock;
    auto t = clock::to_time_t(clock::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return tm.tm_min;
}

bool in_xuni_window() {
    // Mine XUNI only while the pool will accept (:56–:59, :00–:04).
    const int m = local_minute_of_hour();
    return m >= 56 || m < 5;
}

bool in_xuni_submit_window() {
    // Start flushing queued XUNI one minute early (:55), through end of pool window.
    const int m = local_minute_of_hour();
    return m >= 55 || m < 5;
}

bool in_xuni_taper_window() {
    // Last minute of the accept window — reduce XUNI mine intensity, boost submits.
    return local_minute_of_hour() == 4;
}

std::pair<bool, std::string> ready_to_flush(const std::string& block_type) {
    std::string kind = block_type;
    for (char& c : kind) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    // XNM + XBLK: always eligible to submit whenever the network is up.
    // XUNI: flush from :55 so the bag is warm when mining starts at :56.
    if (kind == "XUNI" && !in_xuni_submit_window()) {
        return {false, "waiting_for_xuni_window"};
    }
    if (kind == "XNM" || kind == "XBLK" || kind == "XUNI" || kind == "XEN11" || kind == "NORMAL") {
        return {true, "ready"};
    }
    // Unknown types still flush (better to try than drop).
    return {true, "ready"};
}

int flush_priority(const std::string& block_type) {
    std::string kind = block_type;
    for (char& c : kind) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    // Priority: XBLK > XNM > XUNI (lower = flush sooner). (KIMI)
    if (kind == "XBLK") return 0;
    if (kind == "XNM" || kind == "XEN11" || kind == "NORMAL") return 1;
    if (kind == "XUNI") return 2;
    return 3;
}

}  // namespace xn
