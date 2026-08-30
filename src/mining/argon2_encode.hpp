#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "common.hpp"

namespace xn {

bool is_argon2_encoded(const std::string& hash_str);
// 64 hex chars, no NULs. Garbage GPU/keygen hits must not enter the bag.
bool is_hex64_key(const std::string& key);
std::optional<int> memory_cost_from_hash(const std::string& hash_str);

// Build PHC-encoded Argon2id string from salt + GPU pure base64 digest.
// Matches what the pool expects without re-running full Argon2 on CPU.
std::string encode_argon2id_phc(const std::string& salt_hex, uint32_t memory_cost,
                                uint32_t time_cost, uint32_t parallelism,
                                const std::string& pure_b64_hash);

// Verify pure GPU match is present in encoded form (same check as Python prepare_hit).
std::optional<BlockHit> prepare_hit_for_submit(const BlockHit& hit, const std::string& salt_hex,
                                               int memory_cost, int time_cost = 1,
                                               int parallelism = 1, int hash_len = 64);

}  // namespace xn
