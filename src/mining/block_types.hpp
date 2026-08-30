#pragma once

#include <string>

namespace xn {

inline constexpr int SUPER_UPPERCASE_MIN = 50;

std::string hash_digest_for_superblock(const std::string& hash_str);
int uppercase_count(const std::string& text);
bool is_superblock(const std::string& hash_str, int min_upper = SUPER_UPPERCASE_MIN);
// Returns XUNI, XBLK, XNM, or OTHER.
std::string classify_block(const std::string& hash_str, const std::string& block_type = "");

}  // namespace xn
