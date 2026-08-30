#pragma once

#include <string>
#include <utility>

namespace xn {

// Pool accept window is :56-:59 + :00-:04. We start submit flush at :55 and taper mine at :04.
inline constexpr const char* XUNI_WINDOW_LABEL = "submit :55–:04 · mine :56–:04 · taper :04";
inline constexpr const char* DIFFICULTY_CHANGE_REASON = "difficulty_change";
inline constexpr const char* OUTSIDE_XUNI_WINDOW_REASON = "outside_xuni_window";
inline constexpr const char* SHUTDOWN_PENDING_REASON = "shutdown_pending";
inline constexpr const char* MATCH_WINDOW_NEW_REASON = "match_window_new";

/// Local minute-of-hour [0, 59].
int local_minute_of_hour();

/// Pool XUNI accept window (:56–:59 and :00–:04) — also the CUDA mine window.
bool in_xuni_window();
/// Submit early (:55) through end of pool window so queued XUNI flush before mine starts.
bool in_xuni_submit_window();
/// Last minute of the window (:04) — taper mining, push submits harder.
bool in_xuni_taper_window();

// Ready to flush this block type (network checked by caller).
// XNM and XBLK are always ready; XUNI only inside the submit window.
std::pair<bool, std::string> ready_to_flush(const std::string& block_type);
// Lower = flush sooner (XNM, then XBLK, then XUNI).
int flush_priority(const std::string& block_type);

}  // namespace xn
