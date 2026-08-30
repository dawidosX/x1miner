#include "monitoring/wallet.hpp"

#include "util/http.hpp"
#include "util/paths.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace xn {
namespace {

constexpr const char* XUNI_CONTRACT = "0x999999cf1046e68e36e1aa2e0e07105eddd00002";
constexpr const char* XBLK_CONTRACT = "0x999999cf1046e68e36e1aa2e0e07105eddd00001";
constexpr double WEI_PER_TOKEN = 1e18;
// Primary HTTPS RPC + HTTP twin fallback (API findings §11.4)
constexpr const char* RPC_PRIMARY = "https://xenblocks.io:5556";
constexpr const char* RPC_FALLBACK = "http://xenblocks.io:5555";

std::optional<std::string> rpc_result_url(const char* url, const std::string& method,
                                          const nlohmann::json& params) {
    nlohmann::json body = {{"jsonrpc", "2.0"}, {"method", method}, {"params", params}, {"id", 1}};
    auto resp = http_post_json(url, body.dump(), 12000);
    if (resp.status < 200 || resp.status >= 300) return std::nullopt;
    try {
        auto j = nlohmann::json::parse(resp.body);
        if (!j.contains("result")) return std::nullopt;
        if (j["result"].is_string()) return j["result"].get<std::string>();
        return j["result"].dump();
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::string> rpc_result(const std::string& method, const nlohmann::json& params) {
    if (auto r = rpc_result_url(RPC_PRIMARY, method, params)) return r;
    return rpc_result_url(RPC_FALLBACK, method, params);
}

double hex_wei_to_token(const std::string& hex) {
    // Parse hex integer carefully for large values
    std::string h = hex;
    if (h.rfind("0x", 0) == 0 || h.rfind("0X", 0) == 0) h = h.substr(2);
    if (h.empty()) return 0;
    // Use long double via stepwise
    long double val = 0;
    for (char c : h) {
        int d = 0;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else continue;
        val = val * 16.0L + d;
    }
    return static_cast<double>(val / static_cast<long double>(WEI_PER_TOKEN));
}

}  // namespace

WalletBalanceTracker::WalletBalanceTracker(std::string address, std::filesystem::path history_path)
    : address_(std::move(address)), history_path_(std::move(history_path)) {
    ensure_parent_dir(history_path_);
}

bool WalletBalanceTracker::fetch(TokenBalances& out) {
    auto xnm_hex = rpc_result("eth_getBalance", nlohmann::json::array({address_, "latest"}));
    if (!xnm_hex) return false;
    out.xnm = hex_wei_to_token(*xnm_hex);

    std::string data = "0x70a08231" + std::string(24, '0');
    // pad address without 0x to 64 hex chars
    std::string addr = address_;
    if (addr.size() >= 2) addr = addr.substr(2);
    while (addr.size() < 64) addr = "0" + addr;
    data = "0x70a08231" + addr;

    auto xuni_hex =
        rpc_result("eth_call", nlohmann::json::array({nlohmann::json{{"to", XUNI_CONTRACT}, {"data", data}}, "latest"}));
    auto xblk_hex =
        rpc_result("eth_call", nlohmann::json::array({nlohmann::json{{"to", XBLK_CONTRACT}, {"data", data}}, "latest"}));
    if (xuni_hex) out.xuni = hex_wei_to_token(*xuni_hex);
    if (xblk_hex) out.xblk = hex_wei_to_token(*xblk_hex);
    return true;
}

void WalletBalanceTracker::maybe_refresh(bool force) {
    auto now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
                   .count();
    if (!force && now - last_refresh_ < 120.0) return;
    TokenBalances bal;
    if (!fetch(bal)) return;
    {
        std::lock_guard<std::mutex> lock(mu_);
        current_ = bal;
        last_refresh_ = now;
    }
    try {
        nlohmann::json j = {{"ts", now_iso_local()},
                            {"xnm", bal.xnm},
                            {"xuni", bal.xuni},
                            {"xblk", bal.xblk}};
        std::ofstream out(history_path_, std::ios::app);
        out << j.dump() << "\n";
    } catch (...) {
    }
}

std::optional<TokenBalances> WalletBalanceTracker::current() const {
    std::lock_guard<std::mutex> lock(mu_);
    return current_;
}

std::string WalletBalanceTracker::summary_line() const {
    auto c = current();
    if (!c) return "n/a";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << "XNM " << c->xnm << "   XUNI " << c->xuni
        << "   XBLK " << c->xblk;
    return oss.str();
}

}  // namespace xn
