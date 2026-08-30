#include "monitoring/dashboard.hpp"

#include "common.hpp"
#include "util/paths.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace xn {
namespace {

// ASCII-safe colors (basic 16-color ANSI — works cleanly in Windows Terminal / conhost)
constexpr const char* RST = "\x1b[0m";
constexpr const char* DIM = "\x1b[90m";
constexpr const char* BOLD = "\x1b[1m";
constexpr const char* CYAN = "\x1b[36m";
constexpr const char* GREEN = "\x1b[32m";
constexpr const char* YELLOW = "\x1b[33m";
constexpr const char* RED = "\x1b[31m";
constexpr const char* WHITE = "\x1b[37m";
constexpr const char* CLR_EOL = "\x1b[K";

// Wider panel + generous padding for readability.
constexpr int kWidth = 84;
constexpr int kLabelW = 12;

std::string ascii_clean(std::string s) {
    for (char& c : s) {
        unsigned char u = static_cast<unsigned char>(c);
        if (u < 32 || u > 126) c = '?';
    }
    return s;
}

std::string pad_right(std::string s, int width) {
    if (static_cast<int>(s.size()) < width) s.append(static_cast<size_t>(width - s.size()), ' ');
    if (static_cast<int>(s.size()) > width) s.resize(static_cast<size_t>(width));
    return s;
}

std::string fmt_hps(double hps) {
    std::ostringstream oss;
    oss << std::fixed;
    if (hps >= 1e6) {
        oss << std::setprecision(2) << (hps / 1e6) << " MH/s";
    } else if (hps >= 1e3) {
        oss << std::setprecision(1) << (hps / 1e3) << " kH/s";
    } else {
        oss << std::setprecision(0) << hps << " H/s";
    }
    return oss.str();
}

// Session average XNM found per second (includes queued, not only accepted).
std::string fmt_xnm_ps(double xnm_ps) {
    std::ostringstream oss;
    oss << std::fixed;
    if (xnm_ps >= 100.0)
        oss << std::setprecision(1) << xnm_ps;
    else if (xnm_ps >= 1.0)
        oss << std::setprecision(2) << xnm_ps;
    else if (xnm_ps >= 0.01)
        oss << std::setprecision(3) << xnm_ps;
    else
        oss << std::setprecision(4) << xnm_ps;
    oss << " XNM/s";
    return oss.str();
}

std::string fmt_hashes(int64_t n) {
    std::ostringstream oss;
    if (n >= 1'000'000'000) {
        oss << std::fixed << std::setprecision(2) << (n / 1e9) << "B";
    } else if (n >= 1'000'000) {
        oss << std::fixed << std::setprecision(2) << (n / 1e6) << "M";
    } else if (n >= 1'000) {
        oss << std::fixed << std::setprecision(1) << (n / 1e3) << "k";
    } else {
        oss << n;
    }
    return oss.str();
}

std::string fmt_uptime(int s) {
    int h = s / 3600;
    int m = (s % 3600) / 60;
    int sec = s % 60;
    std::ostringstream oss;
    if (h > 0)
        oss << h << "h " << m << "m";
    else if (m > 0)
        oss << m << "m " << sec << "s";
    else
        oss << sec << "s";
    return oss.str();
}

std::string short_addr(const std::string& a) {
    if (a.size() < 12) return a;
    return a.substr(0, 6) + "..." + a.substr(a.size() - 4);
}

const char* temp_color(int c, int warn, int max) {
    if (c >= max) return RED;
    if (c >= warn) return YELLOW;
    return GREEN;
}

void row(std::ostringstream& oss, const std::string& body) {
    oss << DIM << "|" << RST << body << CLR_EOL << "\n";
}

void rule(std::ostringstream& oss, char edge, char fill) {
    oss << DIM << edge << std::string(static_cast<size_t>(kWidth), fill) << edge << RST << CLR_EOL
        << "\n";
}

void blank_row(std::ostringstream& oss) {
    oss << DIM << "|" << RST << std::string(static_cast<size_t>(kWidth), ' ') << CLR_EOL << "\n";
}

// "  Label     value..." with consistent label column.
std::string labeled(const std::string& label, const std::string& value) {
    return std::string("  ") + pad_right(label, kLabelW) + "  " + value;
}

}  // namespace

MinerDashboard::MinerDashboard(const Settings& settings) : settings_(settings) {}

void MinerDashboard::start() {
    active_ = true;
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h && h != INVALID_HANDLE_VALUE) {
        DWORD mode = 0;
        if (GetConsoleMode(h, &mode)) {
            SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
        }
        // UTF-8 can still break box glyphs; we use pure ASCII UI.
        SetConsoleOutputCP(65001);
    }
#endif
    // Alternate buffer + hide cursor: stable panel, no scroll spam.
    std::cout << "\x1b[?1049h\x1b[?25l\x1b[H\x1b[2J" << std::flush;
}

void MinerDashboard::stop() {
    if (!active_) return;
    active_ = false;
    std::cout << "\x1b[?25h\x1b[?1049l" << RST << std::flush;
}

void MinerDashboard::set_status(const std::string& status) {
    std::lock_guard<std::mutex> lock(mu_);
    status_ = ascii_clean(status);
}

void MinerDashboard::set_network(bool ok, std::optional<int> difficulty, bool stale) {
    std::lock_guard<std::mutex> lock(mu_);
    network_ok_ = ok;
    network_stale_ = stale;
    difficulty_ = difficulty;
}

void MinerDashboard::set_mining_m(int mining_m, bool force_hybrid) {
    std::lock_guard<std::mutex> lock(mu_);
    mining_m_ = mining_m;
    force_hybrid_ = force_hybrid;
}

void MinerDashboard::set_cuda_batch(int batch, int lanes, double thermal_scale) {
    std::lock_guard<std::mutex> lock(mu_);
    cuda_batch_ = batch;
    cuda_lanes_ = lanes;
    thermal_scale_ = thermal_scale;
}

void MinerDashboard::set_wallet_line(const std::string& line) {
    std::lock_guard<std::mutex> lock(mu_);
    wallet_line_ = ascii_clean(line);
}

void MinerDashboard::set_uptime_s(int uptime_s) {
    std::lock_guard<std::mutex> lock(mu_);
    uptime_s_ = uptime_s;
}

void MinerDashboard::event(const std::string& action, const std::string& block,
                           const std::string& detail) {
    std::lock_guard<std::mutex> lock(mu_);
    std::string e = ascii_clean(action + " " + block);
    if (!detail.empty()) e += " " + ascii_clean(detail);
    events_.push_back(std::move(e));
    if (events_.size() > 5) events_.erase(events_.begin());
}

void MinerDashboard::update(const MiningStats& stats, const GpuSnapshot* gpu,
                            const std::unordered_map<std::string, int>& pending_by_type,
                            const std::unordered_map<std::string, int>& /*resubmission*/) {
    std::lock_guard<std::mutex> lock(mu_);
    stats_ = stats;
    if (gpu) gpu_ = *gpu;
    pending_xuni_ = 0;
    pending_xnm_ = 0;
    pending_xblk_ = 0;
    int q = 0;
    for (const auto& kv_pair : pending_by_type) {
        q += kv_pair.second;
        if (kv_pair.first == "XUNI")
            pending_xuni_ = kv_pair.second;
        else if (kv_pair.first == "XBLK")
            pending_xblk_ = kv_pair.second;
        else if (kv_pair.first == "XNM")
            pending_xnm_ = kv_pair.second;
    }
    stats_.queued = q;
}

void MinerDashboard::render() {
    std::lock_guard<std::mutex> lock(mu_);

    // status.json export runs even when the terminal dashboard is off,
    // so the web dashboard (x1watch.xyz) still gets fresh data.
    maybe_export_status_json();

    if (!active_) return;

    std::ostringstream oss;
    oss << "\x1b[H";

    // ---- Header ----
    rule(oss, '+', '=');
    {
        std::ostringstream b;
        b << "  " << BOLD << CYAN << kAppName << RST << "    " << DIM << kAppTagline << RST
          << "    " << DIM << "v" << kMinerVersion << RST;
        row(oss, b.str());
    }
    rule(oss, '+', '=');
    blank_row(oss);

    // ---- Status / wallet ----
    {
        const char* sc = GREEN;
        std::string st = status_.empty() ? "Running" : status_;
        if (st.find("cool") != std::string::npos || st.find("Stop") != std::string::npos ||
            st.find("queue") != std::string::npos || st.find("Hybrid") != std::string::npos)
            sc = YELLOW;
        if (st.find("error") != std::string::npos || st.find("fail") != std::string::npos) sc = RED;
        std::ostringstream val;
        val << sc << st << RST << "      " << DIM << "uptime" << RST << "  " << WHITE
            << fmt_uptime(uptime_s_) << RST;
        row(oss, labeled("Status", val.str()));
    }
    {
        std::ostringstream val;
        val << CYAN << short_addr(settings_.address) << RST << "      " << DIM << "worker" << RST
            << "  " << WHITE << settings_.worker << RST;
        row(oss, labeled("Wallet", val.str()));
    }
    blank_row(oss);

    // ---- Network + CUDA ----
    {
        std::ostringstream val;
        if (network_ok_ && !network_stale_)
            val << GREEN << "[ONLINE]" << RST;
        else if (network_ok_ && network_stale_)
            val << YELLOW << "[STALE]" << RST;
        else
            val << RED << "[DOWN]" << RST;

        if (force_hybrid_ && mining_m_ > 0) {
            val << "      " << BOLD << GREEN << "mine m=" << mining_m_ << RST << DIM << " (fixed)"
                << RST << "      " << WHITE
                << "net m=" << (difficulty_ ? std::to_string(*difficulty_) : "-") << RST;
            if (difficulty_ && *difficulty_ == mining_m_)
                val << "  " << GREEN << "MATCH" << RST;
            else if (difficulty_)
                val << "  " << YELLOW << "queue" << RST;
        } else {
            val << "      " << WHITE << "m=" << (difficulty_ ? std::to_string(*difficulty_) : "-")
                << RST;
        }
        row(oss, labeled("Network", val.str()));
    }
    {
        std::ostringstream val;
        val << WHITE << "batch " << cuda_batch_ << "  x  " << cuda_lanes_ << " lane"
            << (cuda_lanes_ == 1 ? "" : "s") << RST;
        if (thermal_scale_ < 0.995) {
            const int sp = static_cast<int>(thermal_scale_ * 100.0 + 0.5);
            const char* sc = (sp <= 40) ? RED : (sp <= 70) ? YELLOW : GREEN;
            val << "      " << DIM << "auto" << RST << "  " << sc << sp << "%" << RST;
        }
        if (force_hybrid_) val << "      " << DIM << "mode hybrid" << RST;
        row(oss, labeled("CUDA", val.str()));
    }

    blank_row(oss);
    rule(oss, '+', '-');
    blank_row(oss);

    // ---- Performance (split across lines for readability) ----
    {
        const double avg_xnm = stats_.avg_xnm_per_s(uptime_s_);
        std::ostringstream val;
        val << BOLD << WHITE << fmt_hps(stats_.hps_ema) << RST << "      " << BOLD << CYAN
            << fmt_xnm_ps(avg_xnm) << RST << "  " << DIM << "(found / uptime)" << RST;
        row(oss, labeled("Speed", val.str()));
    }
    {
        std::ostringstream val;
        val << WHITE << fmt_hashes(stats_.total_hashes) << RST << "      " << DIM << "found"
            << RST << "  " << WHITE << stats_.found_total() << RST << "      " << DIM << "accept"
            << RST << "  " << GREEN << stats_.accepted_total() << RST << "      " << DIM
            << "reject" << RST << "  " << (stats_.rejected_total() ? RED : DIM)
            << stats_.rejected_total() << RST;
        row(oss, labeled("Hashes", val.str()));
    }
    blank_row(oss);

    // ---- Blocks found / accepted ----
    {
        int ax = stats_.accepted_live_xuni + stats_.accepted_flush_xuni;
        int an = stats_.accepted_xnm_total();
        int ab = stats_.accepted_live_xblk + stats_.accepted_flush_xblk;
        std::ostringstream val;
        val << DIM << "XNM" << RST << "  " << WHITE << stats_.found_xnm << RST << DIM << " / "
            << RST << GREEN << an << RST << "      " << DIM << "XBLK" << RST << "  " << WHITE
            << stats_.found_xblk << RST << DIM << " / " << RST << GREEN << ab << RST << "      "
            << DIM << "XUNI" << RST << "  " << WHITE << stats_.found_xuni << RST << DIM << " / "
            << RST << GREEN << ax << RST << "      " << DIM << "(F / A)" << RST;
        row(oss, labeled("Blocks", val.str()));
    }
    {
        std::ostringstream val;
        val << (stats_.queued ? YELLOW : WHITE) << stats_.queued << RST << "      " << DIM
            << "XNM" << RST << "  " << (pending_xnm_ ? YELLOW : WHITE) << pending_xnm_ << RST
            << "      " << DIM << "XBLK" << RST << "  " << WHITE << pending_xblk_ << RST
            << "      " << DIM << "XUNI" << RST << "  " << WHITE << pending_xuni_ << RST;
        row(oss, labeled("Queue", val.str()));
    }

    blank_row(oss);
    rule(oss, '+', '-');
    blank_row(oss);

    // ---- GPU ----
    if (gpu_) {
        {
            std::ostringstream val;
            val << WHITE << ascii_clean(gpu_->name) << RST;
            row(oss, labeled("GPU", val.str()));
        }
        {
            double vram_pct =
                gpu_->total_mib > 0 ? (100.0 * gpu_->used_mib / gpu_->total_mib) : 0.0;
            std::ostringstream val;
            val << WHITE << gpu_->util_pct << "%" << RST << " util      " << WHITE
                << gpu_->used_mib << "/" << gpu_->total_mib << " MiB" << RST << " (" << std::fixed
                << std::setprecision(0) << vram_pct << "%)";
            if (gpu_->power_w >= 0) {
                val << "      " << WHITE << std::setprecision(0) << gpu_->power_w << " W" << RST;
            }
            row(oss, labeled("Load", val.str()));
        }
        {
            const char* gc =
                temp_color(gpu_->temperature_c, settings_.warn_gpu_temp_c, settings_.max_gpu_temp_c);
            std::ostringstream val;
            val << DIM << "GPU" << RST << "  " << gc << gpu_->temperature_c << " C" << RST;
            if (gpu_->has_memory_junction()) {
                const char* mc = temp_color(gpu_->memory_junction_c, settings_.warn_mem_temp_c,
                                            settings_.max_mem_temp_c);
                val << "      " << DIM << "Mem junc" << RST << "  " << mc << gpu_->memory_junction_c
                    << " C" << RST << DIM << " / cap " << settings_.max_mem_temp_c << RST;
            } else {
                val << "      " << DIM << "Mem junc  -" << RST;
            }
            row(oss, labeled("Temp", val.str()));
        }
    } else {
        row(oss, labeled("GPU", std::string(DIM) + "-" + RST));
        blank_row(oss);
    }
    blank_row(oss);
    {
        std::ostringstream val;
        val << (wallet_line_.empty() ? std::string(DIM) + "-" + RST : wallet_line_);
        row(oss, labeled("Holdings", val.str()));
    }

    blank_row(oss);
    rule(oss, '+', '-');
    blank_row(oss);
    row(oss, labeled("Recent", std::string(DIM) + "latest events" + RST));
    blank_row(oss);

    std::vector<std::string> recent;
    for (auto it = events_.rbegin(); it != events_.rend() && recent.size() < 5; ++it) {
        recent.push_back(*it);
    }
    while (recent.size() < 5) recent.push_back("");

    for (const auto& ev : recent) {
        if (ev.empty()) {
            row(oss, std::string("                ") + DIM + "-" + RST);
            continue;
        }
        const char* ec = WHITE;
        if (ev.find("ACCEPT") != std::string::npos || ev.find("FOUND") != std::string::npos ||
            ev.find("SUBMIT") != std::string::npos)
            ec = GREEN;
        else if (ev.find("QUEUE") != std::string::npos || ev.find("WARN") != std::string::npos ||
                 ev.find("Hybrid") != std::string::npos)
            ec = YELLOW;
        else if (ev.find("REJECT") != std::string::npos || ev.find("FAIL") != std::string::npos ||
                 ev.find("ERROR") != std::string::npos)
            ec = RED;
        std::string show = ev.size() > 66 ? ev.substr(0, 63) + "..." : ev;
        row(oss, std::string("                ") + ec + show + RST);
    }

    blank_row(oss);
    rule(oss, '+', '=');
    {
        std::ostringstream b;
        b << DIM << "  Ctrl+C or close window  =  stop and bag queue for next start" << RST;
        oss << b.str() << CLR_EOL << "\n";
    }
    oss << "\x1b[J";
    std::cout << oss.str() << std::flush;
}

void MinerDashboard::maybe_export_status_json() {
    const double now =
        std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now - last_json_export_at_ < 4.0) return;  // export every ~4 s
    last_json_export_at_ = now;

    nlohmann::json j;
    j["rig_name"] = settings_.worker;
    j["version"] = kMinerVersion;
    j["timestamp"] = now_iso_local();
    j["uptime_s"] = uptime_s_;
    j["hashrate"] = stats_.hps_ema;
    j["mining_m"] = mining_m_;
    j["network_m"] = difficulty_.value_or(0);
    j["network_ok"] = network_ok_;
    j["found"] = stats_.found_total();
    j["accept"] = stats_.accepted_total();
    j["reject"] = stats_.rejected_total();
    j["queued"] = stats_.queued;
    j["blocks_xnm"] = stats_.found_xnm;
    j["blocks_xblk"] = stats_.found_xblk;
    j["blocks_xuni"] = stats_.found_xuni;
    j["pending_xnm"] = pending_xnm_;
    j["pending_xblk"] = pending_xblk_;
    j["pending_xuni"] = pending_xuni_;
    j["cuda_batch"] = cuda_batch_;
    j["cuda_lanes"] = cuda_lanes_;
    j["thermal_scale"] = thermal_scale_;

    if (gpu_) {
        j["gpu_name"] = gpu_->name;
        j["gpu_util"] = gpu_->util_pct;
        j["gpu_temp_c"] = gpu_->temperature_c;
        j["gpu_mem_junc_c"] = gpu_->memory_junction_c;
        j["vram_used_mib"] = gpu_->used_mib;
        j["vram_total_mib"] = gpu_->total_mib;
        j["power_w"] = gpu_->power_w;
    }

    const std::filesystem::path out_path = settings_.root / "data" / "status.json";
    std::error_code ec;
    std::filesystem::create_directories(out_path.parent_path(), ec);
    // Atomic-ish write: tmp + rename, so a web reader never sees a partial file.
    const std::filesystem::path tmp_path = out_path.parent_path() / "status.json.tmp";
    {
        std::ofstream ofs(tmp_path, std::ios::trunc);
        if (!ofs) return;
        ofs << j.dump(2) << "\n";
    }
    std::filesystem::rename(tmp_path, out_path, ec);
    if (ec) {
        std::filesystem::remove(out_path, ec);
        ec.clear();
        std::filesystem::rename(tmp_path, out_path, ec);
    }
}

}  // namespace xn
