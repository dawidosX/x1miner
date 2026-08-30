#pragma once

#include <cstddef>
#include <string>

namespace hashapi {

/// Persistent keygen workers. Thread count is chosen from CPU cores (never more
/// threads than logical CPUs). All GPU lanes share this pool.
void configureKeygenPool(int threads);
int keygenPoolThreads();
/// Fill `count` keys of `key_length` hex chars into a flat arena (count * key_length).
void keygenFillFlat(char* arena, std::size_t count, std::size_t key_length,
                    const std::string& prefix);

}  // namespace hashapi
