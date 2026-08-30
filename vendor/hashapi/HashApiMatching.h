#pragma once

#include "HashApiTypes.h"

#include <cstddef>
#include <string>

namespace hashapi {

bool isSuperblockHash(const std::string& hash);
bool isSuperblockHash(const char* hash, std::size_t hash_len);
bool hasXuniMatch(const std::string& hash);
bool hasXuniMatch(const char* hash, std::size_t hash_len);
bool containsXen11(const char* hash, std::size_t hash_len);

void appendMatches(const HashApiRequest& request,
                   HashApiResult& result,
                   const std::string& key,
                   const std::string& hash,
                   std::size_t attempt_index);

/// Hot path: key is only materialized when a pattern matches.
void appendMatchesRaw(const HashApiRequest& request,
                      HashApiResult& result,
                      const char* key,
                      std::size_t key_len,
                      const char* hash,
                      std::size_t hash_len,
                      std::size_t attempt_index);

} // namespace hashapi
