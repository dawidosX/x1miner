#pragma once

#include <filesystem>
#include <string>

namespace xn {

std::filesystem::path resolve_path(const std::filesystem::path& root, const std::string& rel);
std::string now_iso_local();
std::string now_iso_utc();
void ensure_parent_dir(const std::filesystem::path& path);

}  // namespace xn
