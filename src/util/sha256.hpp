#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace xn {

// Portable SHA-256. Returns lowercase hex, optionally truncated to prefix_bytes.
std::string sha256_hex(const std::string& data, std::size_t prefix_bytes = 32);

}  // namespace xn
