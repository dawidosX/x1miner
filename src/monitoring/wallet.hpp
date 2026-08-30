#pragma once

#include <filesystem>
#include <mutex>
#include <optional>
#include <string>

namespace xn {

struct TokenBalances {
    double xnm = 0;
    double xuni = 0;
    double xblk = 0;
};

class WalletBalanceTracker {
public:
    WalletBalanceTracker(std::string address, std::filesystem::path history_path);

    void maybe_refresh(bool force = false);
    std::optional<TokenBalances> current() const;
    std::string summary_line() const;

private:
    bool fetch(TokenBalances& out);

    std::string address_;
    std::filesystem::path history_path_;
    mutable std::mutex mu_;
    std::optional<TokenBalances> current_;
    double last_refresh_ = 0;
};

}  // namespace xn
