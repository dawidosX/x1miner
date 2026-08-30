#include "HashApiMatching.h"

#include <algorithm>
#include <cctype>

namespace hashapi {

bool isSuperblockHash(const char* hash, std::size_t hash_len)
{
    if (hash == nullptr || hash_len == 0) {
        return false;
    }
    std::size_t uppercase_count = 0;
    for (std::size_t i = 0; i < hash_len; ++i) {
        if (std::isupper(static_cast<unsigned char>(hash[i])) != 0) {
            ++uppercase_count;
        }
    }
    return uppercase_count >= 50;
}

bool isSuperblockHash(const std::string& hash)
{
    return isSuperblockHash(hash.data(), hash.size());
}

bool hasXuniMatch(const char* hash, std::size_t hash_len)
{
    if (hash == nullptr || hash_len < 5) {
        return false;
    }
    for (std::size_t i = 0; i + 4 < hash_len; ++i) {
        if (hash[i] == 'X' && hash[i + 1] == 'U' && hash[i + 2] == 'N' && hash[i + 3] == 'I' &&
            std::isdigit(static_cast<unsigned char>(hash[i + 4])) != 0) {
            return true;
        }
    }
    return false;
}

bool hasXuniMatch(const std::string& hash)
{
    return hasXuniMatch(hash.data(), hash.size());
}

bool containsXen11(const char* hash, std::size_t hash_len)
{
    if (hash == nullptr || hash_len < 5) {
        return false;
    }
    const char* end = hash + (hash_len - 4);
    for (const char* p = hash; p < end; ++p) {
        if (p[0] == 'X' && p[1] == 'E' && p[2] == 'N' && p[3] == '1' && p[4] == '1') {
            return true;
        }
    }
    return false;
}

namespace {

bool containsPatternRaw(const char* hash, std::size_t hash_len, const std::string& pattern)
{
    if (pattern.size() == 5 &&
        pattern[0] == 'X' && pattern[1] == 'E' && pattern[2] == 'N' &&
        pattern[3] == '1' && pattern[4] == '1') {
        return containsXen11(hash, hash_len);
    }
    if (hash == nullptr || pattern.empty() || hash_len < pattern.size()) {
        return false;
    }
    return std::search(hash, hash + hash_len, pattern.begin(), pattern.end()) != hash + hash_len;
}

} // namespace

void appendMatches(const HashApiRequest& request,
                   HashApiResult& result,
                   const std::string& key,
                   const std::string& hash,
                   std::size_t attempt_index)
{
    appendMatchesRaw(request, result, key.data(), key.size(), hash.data(), hash.size(),
                     attempt_index);
}

namespace {

bool isHex64KeyRaw(const char* key, std::size_t key_len)
{
    if (key == nullptr || key_len != 64) return false;
    for (std::size_t i = 0; i < 64; ++i) {
        const unsigned char c = static_cast<unsigned char>(key[i]);
        if (std::isxdigit(c) == 0) return false;
    }
    return true;
}

}  // namespace

void appendMatchesRaw(const HashApiRequest& request,
                      HashApiResult& result,
                      const char* key,
                      std::size_t key_len,
                      const char* hash,
                      std::size_t hash_len,
                      std::size_t attempt_index)
{
    // Never pair a digest with a missing/garbage key (4x2 snapshot of the
    // wrong slot produced all-zero keys that still had XEN11 in the hash).
    if (!isHex64KeyRaw(key, key_len)) {
        return;
    }
    // Fast reject: scan the encoded digest only; XEN11 must be 5 in a row.
    const bool xen = containsPatternRaw(hash, hash_len, request.target_pattern);
    const bool xuni = request.allow_xuni && hasXuniMatch(hash, hash_len);
    if (!xen && !xuni) {
        return;
    }

    std::string key_str(key, key_len);
    std::string hash_str(hash, hash_len);

    if (xen) {
        result.matches.push_back({
            key_str,
            hash_str,
            request.target_pattern,
            attempt_index,
            isSuperblockHash(hash, hash_len),
        });
    }
    if (xuni) {
        result.matches.push_back({
            std::move(key_str),
            std::move(hash_str),
            "XUNI",
            attempt_index,
            false,
        });
    }
}

} // namespace hashapi
