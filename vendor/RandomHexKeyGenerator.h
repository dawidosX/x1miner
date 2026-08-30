#pragma once

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

class RandomHexKeyGenerator {
public:
    RandomHexKeyGenerator(const std::string& initial_prefix = "", size_t key_length = 64)
        : total_length(key_length) {
            setPrefix(initial_prefix);
            std::random_device rd;
            auto seed = rd() ^ static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
            generator.seed(seed);
        }

    void setPrefix(const std::string& new_prefix) {
        prefix = new_prefix;
        std::transform(prefix.begin(), prefix.end(), prefix.begin(),
                       [](unsigned char c){ return std::tolower(c); });
    }

    std::string nextRandomKey() {
        std::string key;
        key.resize(total_length);
        fillKey(key.data(), total_length);
        return key;
    }

    // Write one key into an existing buffer (size must be total_length).
    void fillInto(std::string& key) {
        if (key.size() != total_length) {
            key.assign(total_length, '0');
        }
        fillKey(&key[0], total_length);
    }

    // Write one key into a preallocated char buffer (no heap traffic).
    void fillInto(char* dst, std::size_t dst_len) {
        if (dst == nullptr || dst_len < total_length) return;
        fillKey(dst, total_length);
    }

    // Bulk-generate into a flat arena: count * total_length contiguous bytes.
    // splitmix64 — mining keys are not a crypto RNG; mt19937 here starved the GPU.
    void fillMany(char* arena, std::size_t count) {
        if (arena == nullptr || count == 0) return;
        const size_t pre = std::min(prefix.size(), total_length);
        uint64_t s = (static_cast<uint64_t>(generator()) << 32) ^
                     static_cast<uint64_t>(generator()) ^
                     (static_cast<uint64_t>(count) * 0x9E3779B97F4A7C15ULL);
        for (std::size_t i = 0; i < count; ++i) {
            char* dst = arena + i * total_length;
            if (pre) std::memcpy(dst, prefix.data(), pre);
            size_t pos = pre;
            while (pos < total_length) {
                s += 0x9E3779B97F4A7C15ULL;
                uint64_t z = s;
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
                z ^= z >> 31;
                for (int n = 0; n < 16 && pos < total_length; ++n) {
                    dst[pos++] = kHexChars[z & 0x0f];
                    z >>= 4;
                }
            }
        }
    }

    // Bulk-generate into pre-sized strings (avoids per-key reallocation thrash).
    void fillKeys(std::vector<std::string>& out, std::size_t count) {
        out.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            fillInto(out[i]);
        }
    }

    std::size_t keyLength() const { return total_length; }

private:
    inline static constexpr char kHexChars[] = "0123456789abcdef";
    std::string prefix;
    size_t total_length;
    std::mt19937 generator;

    void fillKey(char* dst, size_t len) {
        const size_t pre = std::min(prefix.size(), len);
        if (pre) {
            std::memcpy(dst, prefix.data(), pre);
        }
        size_t pos = pre;
        while (pos < len) {
            // Draw 64 bits at a time (2x mt19937) for fewer generator calls.
            const std::uint64_t bits =
                (static_cast<std::uint64_t>(generator()) << 32) |
                static_cast<std::uint64_t>(generator());
            std::uint64_t x = bits;
            for (int n = 0; n < 16 && pos < len; ++n) {
                dst[pos++] = kHexChars[x & 0x0f];
                x >>= 4;
            }
        }
    }
};
