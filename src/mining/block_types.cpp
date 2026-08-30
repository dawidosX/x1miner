#include "mining/block_types.hpp"

#include <algorithm>
#include <cctype>
#include <regex>

namespace xn {

std::string hash_digest_for_superblock(const std::string& hash_str) {
    if (hash_str.empty()) return {};
    if (hash_str.front() == '$' || std::count(hash_str.begin(), hash_str.end(), '$') >= 4) {
        auto pos = hash_str.rfind('$');
        if (pos != std::string::npos) return hash_str.substr(pos + 1);
    }
    return hash_str;
}

int uppercase_count(const std::string& text) {
    int n = 0;
    for (unsigned char ch : text) {
        if (std::isupper(ch)) ++n;
    }
    return n;
}

bool is_superblock(const std::string& hash_str, int min_upper) {
    return uppercase_count(hash_digest_for_superblock(hash_str)) >= min_upper;
}

std::string classify_block(const std::string& hash_str, const std::string& block_type) {
    std::string bt = block_type;
    for (char& c : bt) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    // strip nulls
    bt.erase(std::remove(bt.begin(), bt.end(), '\0'), bt.end());
    while (!bt.empty() && std::isspace(static_cast<unsigned char>(bt.front()))) bt.erase(bt.begin());
    while (!bt.empty() && std::isspace(static_cast<unsigned char>(bt.back()))) bt.pop_back();

    static const std::regex xuni_re("XUNI[0-9]");
    if (bt == "XUNI" || std::regex_search(hash_str, xuni_re)) return "XUNI";
    // Superblock (XBLK) before generic XNM — both are fully queueable/submittable.
    if (bt == "XBLK" || (hash_str.find("XEN11") != std::string::npos && is_superblock(hash_str)))
        return "XBLK";
    if (hash_str.find("XEN11") != std::string::npos || bt == "XEN11" || bt == "NORMAL" ||
        bt == "XNM") {
        return "XNM";
    }
    return "OTHER";
}

}  // namespace xn
