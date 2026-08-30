#include "HashApiEncoding.h"

namespace hashapi {
namespace {

constexpr char kBase64Chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

} // namespace

std::size_t base64EncodedLength(std::size_t in_len)
{
    const std::size_t full_groups = in_len / 3;
    const std::size_t remaining = in_len % 3;
    return full_groups * 4 + (remaining == 0 ? 0 : remaining + 1);
}

std::size_t base64EncodeTo(char* out, std::size_t out_cap, const std::uint8_t* bytes_to_encode,
                           std::size_t in_len)
{
    if (out == nullptr || bytes_to_encode == nullptr) {
        return 0;
    }
    const std::size_t out_len = base64EncodedLength(in_len);
    if (out_cap < out_len) {
        return 0;
    }
    std::size_t w = 0;
    std::size_t offset = 0;
    while (offset + 2 < in_len) {
        const std::uint32_t value =
            (static_cast<std::uint32_t>(bytes_to_encode[offset]) << 16) |
            (static_cast<std::uint32_t>(bytes_to_encode[offset + 1]) << 8) |
            static_cast<std::uint32_t>(bytes_to_encode[offset + 2]);
        out[w++] = kBase64Chars[(value >> 18) & 0x3f];
        out[w++] = kBase64Chars[(value >> 12) & 0x3f];
        out[w++] = kBase64Chars[(value >> 6) & 0x3f];
        out[w++] = kBase64Chars[value & 0x3f];
        offset += 3;
    }

    const std::size_t remaining = in_len - offset;
    if (remaining == 1) {
        const std::uint32_t value = static_cast<std::uint32_t>(bytes_to_encode[offset]) << 16;
        out[w++] = kBase64Chars[(value >> 18) & 0x3f];
        out[w++] = kBase64Chars[(value >> 12) & 0x3f];
    } else if (remaining == 2) {
        const std::uint32_t value =
            (static_cast<std::uint32_t>(bytes_to_encode[offset]) << 16) |
            (static_cast<std::uint32_t>(bytes_to_encode[offset + 1]) << 8);
        out[w++] = kBase64Chars[(value >> 18) & 0x3f];
        out[w++] = kBase64Chars[(value >> 12) & 0x3f];
        out[w++] = kBase64Chars[(value >> 6) & 0x3f];
    }
    return w;
}

void base64EncodeInto(std::string& encoded, const std::uint8_t* bytes_to_encode, std::size_t in_len)
{
    const std::size_t out_len = base64EncodedLength(in_len);
    encoded.assign(out_len, '\0');
    base64EncodeTo(encoded.data(), encoded.size(), bytes_to_encode, in_len);
}

std::string base64Encode(const std::uint8_t* bytes_to_encode, std::size_t in_len)
{
    std::string encoded;
    base64EncodeInto(encoded, bytes_to_encode, in_len);
    return encoded;
}

} // namespace hashapi
