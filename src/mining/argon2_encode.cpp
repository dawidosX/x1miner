#include "mining/argon2_encode.hpp"

#include "hashapi/HashApiEncoding.h"
#include "mining/block_types.hpp"

#include <cctype>
#include <regex>
#include <vector>

namespace xn {
namespace {

std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        int hi = nibble(hex[i]);
        int lo = nibble(hex[i + 1]);
        if (hi < 0 || lo < 0) break;
        out.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return out;
}

}  // namespace

bool is_hex64_key(const std::string& key) {
    if (key.size() != 64) return false;
    for (unsigned char c : key) {
        if (!std::isxdigit(c)) return false;
    }
    return true;
}

bool is_argon2_encoded(const std::string& hash_str) {
    return hash_str.rfind("$argon2", 0) == 0;
}

std::optional<int> memory_cost_from_hash(const std::string& hash_str) {
    if (!is_argon2_encoded(hash_str)) return std::nullopt;
    static const std::regex re(R"(\$m=(\d+),)");
    std::smatch m;
    if (!std::regex_search(hash_str, m, re)) return std::nullopt;
    try {
        return std::stoi(m[1].str());
    } catch (...) {
        return std::nullopt;
    }
}

std::string encode_argon2id_phc(const std::string& salt_hex, uint32_t memory_cost,
                                uint32_t time_cost, uint32_t parallelism,
                                const std::string& pure_b64_hash) {
    auto salt = hex_to_bytes(salt_hex);
    std::string salt_b64 = hashapi::base64Encode(salt.data(), salt.size());
    // PHC: $argon2id$v=19$m=M,t=T,p=P$salt$hash
    std::string out = "$argon2id$v=19$m=";
    out += std::to_string(memory_cost);
    out += ",t=";
    out += std::to_string(time_cost);
    out += ",p=";
    out += std::to_string(parallelism);
    out += "$";
    out += salt_b64;
    out += "$";
    out += pure_b64_hash;
    return out;
}

std::optional<BlockHit> prepare_hit_for_submit(const BlockHit& hit, const std::string& salt_hex,
                                               int memory_cost, int time_cost, int parallelism,
                                               int /*hash_len*/) {
    if (is_argon2_encoded(hit.hash_str)) {
        auto block_m = memory_cost_from_hash(hit.hash_str);
        if (block_m && *block_m != memory_cost) return std::nullopt;
        BlockHit out = hit;
        out.block_type = classify_block(hit.hash_str, hit.block_type);
        return out;
    }

    // GPU returns pure argon2-style base64 digest. Assemble PHC and require
    // the pure digest to appear (same filter as the Python miner).
    std::string encoded =
        encode_argon2id_phc(salt_hex, static_cast<uint32_t>(memory_cost),
                            static_cast<uint32_t>(time_cost),
                            static_cast<uint32_t>(parallelism), hit.hash_str);
    if (encoded.find(hit.hash_str) == std::string::npos) return std::nullopt;

    BlockHit out = hit;
    out.hash_str = std::move(encoded);
    out.block_type = classify_block(out.hash_str, hit.block_type);
    out.memory_cost = memory_cost;
    return out;
}

}  // namespace xn
