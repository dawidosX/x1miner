#include "monitoring/woodyminer.hpp"

#include "common.hpp"
#include "util/http.hpp"

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#include <iphlpapi.h>
#pragma comment(lib, "iphlpapi.lib")
#include <wincrypt.h>
#pragma comment(lib, "bcrypt.lib")
#include <bcrypt.h>
#else
#include "util/sha256.hpp"
#include <cstring>
#include <ifaddrs.h>
#include <net/if.h>
#include <netpacket/packet.h>
#include <sys/socket.h>
#endif

#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>

namespace xn {
namespace {

std::string sha256_hex_prefix(const std::string& data, size_t n = 16) {
#ifdef _WIN32
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD obj_len = 0, data_len = 0, hash_len = 0;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) return "unknown";
    BCryptGetProperty(alg, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&obj_len), sizeof(obj_len),
                      &data_len, 0);
    BCryptGetProperty(alg, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hash_len), sizeof(hash_len),
                      &data_len, 0);
    std::vector<UCHAR> obj(obj_len), digest(hash_len);
    if (BCryptCreateHash(alg, &hash, obj.data(), obj_len, nullptr, 0, 0) != 0) {
        BCryptCloseAlgorithmProvider(alg, 0);
        return "unknown";
    }
    BCryptHashData(hash, reinterpret_cast<PUCHAR>(const_cast<char*>(data.data())),
                   static_cast<ULONG>(data.size()), 0);
    BCryptFinishHash(hash, digest.data(), hash_len, 0);
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(alg, 0);
    std::ostringstream oss;
    for (size_t i = 0; i < digest.size() && oss.str().size() < n * 2; ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[i]);
    }
    auto full = oss.str();
    return full.substr(0, n);
#else
    return sha256_hex(data, n);
#endif
}

#ifdef _WIN32
std::string mac_string() {
    IP_ADAPTER_INFO info[16];
    DWORD len = sizeof(info);
    if (GetAdaptersInfo(info, &len) != ERROR_SUCCESS) return "00:00:00:00:00:00";
    auto* p = info;
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (UINT i = 0; i < p->AddressLength; ++i) {
        if (i) oss << ":";
        oss << std::setw(2) << static_cast<int>(p->Address[i]);
    }
    return oss.str();
}
#else
bool skip_iface(const char* name) {
    if (!name || !*name) return true;
    static const char* kSkip[] = {"lo", "docker", "br-", "veth", "virbr", "cni", "flannel", "tun",
                                  "tap", "wg"};
    for (const char* p : kSkip) {
        if (std::strncmp(name, p, std::strlen(p)) == 0) return true;
    }
    return false;
}

std::string mac_string() {
    ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) != 0 || !ifaddr) return "00:00:00:00:00:00";
    std::string best;
    for (ifaddrs* ifa = ifaddr; ifa; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_PACKET) continue;
        if (skip_iface(ifa->ifa_name)) continue;
        if (!(ifa->ifa_flags & IFF_UP) || (ifa->ifa_flags & IFF_LOOPBACK)) continue;
        auto* sll = reinterpret_cast<sockaddr_ll*>(ifa->ifa_addr);
        if (sll->sll_halen < 6) continue;
        bool zero = true;
        for (int i = 0; i < 6; ++i) {
            if (sll->sll_addr[i] != 0) {
                zero = false;
                break;
            }
        }
        if (zero) continue;
        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (int i = 0; i < 6; ++i) {
            if (i) oss << ":";
            oss << std::setw(2) << static_cast<int>(sll->sll_addr[i]);
        }
        best = oss.str();
        break;
    }
    freeifaddrs(ifaddr);
    return best.empty() ? "00:00:00:00:00:00" : best;
}
#endif

}  // namespace

std::string derive_machine_id(int device_index) {
    std::string mac = mac_string();
    // Match Python: sha256(mac + f"{device_index},")[:16]
    std::string material = mac + std::to_string(device_index) + ",";
    return sha256_hex_prefix(material, 16);
}

WoodyminerUploader::WoodyminerUploader(std::string upload_url, int period_s, std::string custom_name,
                                       std::string miner_address, std::string machine_id,
                                       std::function<MiningStats()> get_stats,
                                       std::function<std::optional<GpuSnapshot>()> get_gpu,
                                       std::function<int()> get_difficulty, double session_started_at,
                                       SessionLogger* logger)
    : upload_url_(std::move(upload_url)),
      period_s_(std::max(15, period_s)),
      custom_name_(std::move(custom_name)),
      miner_address_(std::move(miner_address)),
      machine_id_(std::move(machine_id)),
      get_stats_(std::move(get_stats)),
      get_gpu_(std::move(get_gpu)),
      get_difficulty_(std::move(get_difficulty)),
      session_started_at_(session_started_at),
      logger_(logger) {}

void WoodyminerUploader::start() {
    if (running_) return;
    running_ = true;
    thread_ = std::thread([this] { loop(); });
    if (logger_) {
        logger_->info("Woodyminer leaderboard upload enabled (machineId=" + machine_id_ +
                      ", every " + std::to_string(period_s_) + "s)");
    }
}

void WoodyminerUploader::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void WoodyminerUploader::loop() {
    while (running_) {
        try {
            auto stats = get_stats_();
            auto gpu = get_gpu_ ? get_gpu_() : std::nullopt;
            // Public difficulty = live network m= only. 0 from caller means N/A
            // (hybrid force-mine m= must never be uploaded).
            int difficulty = get_difficulty_ ? get_difficulty_() : 0;
            auto now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
                           .count();
            int uptime = static_cast<int>(std::max(0.0, now - session_started_at_));

            nlohmann::json gpus = nlohmann::json::array();
            int total_power_mw = 0;
            if (gpu) {
                double using_pct = gpu->total_mib > 0 ? (100.0 * gpu->used_mib / gpu->total_mib) : 0.0;
                int power_mw = gpu->power_w >= 0 ? static_cast<int>(gpu->power_w * 1000) : -1;
                if (power_mw >= 0) total_power_mw = power_mw;
                std::ostringstream hr;
                hr << std::fixed << std::setprecision(2) << stats.hps_ema;
                std::ostringstream um;
                um << std::fixed << std::setprecision(1) << using_pct;
                gpus.push_back({{"index", gpu->index},
                                {"name", gpu->name},
                                {"hashrate", hr.str()},
                                {"memory", gpu->total_mib},
                                {"power", power_mw},
                                {"utiliz", gpu->util_pct},
                                {"usingMemory", um.str()},
                                {"hashCount", stats.total_hashes}});
            }

            int normal = stats.accepted_live_xnm + stats.accepted_flush_xnm;
            int super_b = stats.accepted_live_xblk + stats.accepted_flush_xblk;
            std::ostringstream thr;
            thr << std::fixed << std::setprecision(2) << stats.hps_ema;

            // Match the champ miner payload exactly. Woodyminer requires minerAddr
            // to be a 0x wallet (400 if not). Display name goes in customName only.
            nlohmann::json payload = {{"machineId", machine_id_},
                                      {"minerAddr", miner_address_},
                                      {"totalHashrate", thr.str()},
                                      {"totalHashCount", stats.total_hashes},
                                      {"totalPower", total_power_mw},
                                      {"gpus", gpus},
                                      {"uptime", uptime},
                                      {"acceptedBlocks", normal + super_b},
                                      {"normalBlocks", normal},
                                      {"superBlocks", super_b},
                                      {"rejectedBlocks", stats.rejected_total()},
                                      {"version", kMinerVersion}};
            // Network difficulty only; null = N/A (do not invent mine m=).
            if (difficulty > 0) {
                payload["difficulty"] = difficulty;
            } else {
                payload["difficulty"] = nullptr;
            }
            if (!custom_name_.empty()) payload["customName"] = custom_name_;

            auto resp = http_post_json(upload_url_, payload.dump(), 3000, "xenblocksMiner/1.4.0");
            if (logger_ && (resp.status < 200 || resp.status >= 300)) {
                logger_->warn("Woodyminer upload HTTP " + std::to_string(resp.status) +
                              (resp.error.empty() ? "" : (" — " + resp.error)) +
                              (resp.body.empty() ? "" : (" body=" + resp.body.substr(0, 200))));
            }
        } catch (const std::exception& ex) {
            if (logger_) logger_->warn(std::string("Woodyminer upload failed: ") + ex.what());
        } catch (...) {
            if (logger_) logger_->warn("Woodyminer upload failed");
        }

        for (int i = 0; i < period_s_ * 10 && running_; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

}  // namespace xn
