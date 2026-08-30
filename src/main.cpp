#include "app/supervisor.hpp"
#include "common.hpp"
#include "config/settings.hpp"
#include "util/cpu.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace {
std::atomic<xn::Supervisor*> g_supervisor{nullptr};

void handle_signal(int) {
    if (auto* s = g_supervisor.load()) {
        s->request_stop();
        s->persist_queue_for_restart();
    }
}

#ifdef _WIN32
BOOL WINAPI console_handler(DWORD type) {
    // Close / logoff / shutdown can kill the process soon after this returns —
    // bag unsubmitted blocks to disk immediately (no network wait).
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT ||
        type == CTRL_LOGOFF_EVENT || type == CTRL_SHUTDOWN_EVENT) {
        if (auto* s = g_supervisor.load()) {
            s->request_stop();
            s->persist_queue_for_restart();
            // Give main loop a moment to finish clean teardown when user hits Ctrl+C.
            if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT) {
                for (int i = 0; i < 50 && !s->shutdown_complete(); ++i) {
                    Sleep(100);
                }
            }
        }
        return TRUE;
    }
    return FALSE;
}
#endif
}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path config;
    bool no_dashboard = false;
    bool diagnose = false;
    std::optional<int> max_seconds;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config = argv[++i];
        } else if (a == "--no-dashboard") {
            no_dashboard = true;
        } else if (a == "--diagnose") {
            diagnose = true;
        } else if (a == "--max-seconds" && i + 1 < argc) {
            max_seconds = std::stoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            std::cout
                << xn::kAppName << "\n"
                << "Usage: xnminer [--config miner.ini] [--no-dashboard] [--diagnose] "
                   "[--max-seconds N]\n";
            return 0;
        }
    }

    // Resolve root = directory of executable when possible, else cwd
    std::filesystem::path root = std::filesystem::current_path();
#ifdef _WIN32
    wchar_t module_path[MAX_PATH];
    if (GetModuleFileNameW(nullptr, module_path, MAX_PATH) > 0) {
        root = std::filesystem::path(module_path).parent_path();
        auto candidate = root;
        if (!std::filesystem::exists(candidate / "miner.ini") &&
            !std::filesystem::exists(candidate / "miner.ini.example")) {
            if (std::filesystem::exists(root / ".." / ".." / "miner.ini.example")) {
                candidate = (root / ".." / "..").lexically_normal();
            }
        }
        root = candidate;
    }
#elif defined(__linux__)
    char module_path[4096];
    const ssize_t n = ::readlink("/proc/self/exe", module_path, sizeof(module_path) - 1);
    if (n > 0) {
        module_path[n] = '\0';
        root = std::filesystem::path(module_path).parent_path();
        auto candidate = root;
        if (!std::filesystem::exists(candidate / "miner.ini") &&
            !std::filesystem::exists(candidate / "miner.ini.example")) {
            if (std::filesystem::exists(root / ".." / ".." / "miner.ini.example")) {
                candidate = (root / ".." / "..").lexically_normal();
            }
        }
        root = candidate;
    }
#endif

    if (config.empty()) config = root / "miner.ini";

    // Ensure we work relative to project root for data/
    std::error_code ec;
    std::filesystem::current_path(config.parent_path(), ec);

    if (!diagnose) {
        if (!xn::ensure_wallet_configured(config, true)) {
            std::cerr << "Wallet setup required. Edit miner.ini or re-run interactively.\n";
            return 1;
        }
    } else if (!std::filesystem::exists(config)) {
        auto example = config.parent_path() / "miner.ini.example";
        if (std::filesystem::exists(example)) {
            std::filesystem::copy_file(example, config);
        }
    }

    auto settings = xn::load_settings(config);

    // Must run BEFORE NvmlMonitor / any other CUDA call or flags are ignored.
    {
        cudaError_t fl = cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost);
        cudaError_t sd = cudaSetDevice(settings.device_id);
        cudaDeviceProp prop{};
        if (sd == cudaSuccess && cudaGetDeviceProperties(&prop, settings.device_id) == cudaSuccess) {
            const int logical = xn::cpu_logical_count();
            const int physical = xn::cpu_physical_count();
            std::cout << "GPU  " << prop.name << "  sm_" << prop.major << prop.minor
                      << "  VRAM=" << (prop.totalGlobalMem / (1024 * 1024)) << "MiB"
                      << "  L2=" << (prop.l2CacheSize / (1024 * 1024)) << "MiB"
                      << "  flags=" << (fl == cudaSuccess ? "spin+maphost" : cudaGetErrorString(fl))
                      << "\n";
            std::cout << "CPU  " << physical << " cores / " << logical << " threads"
                      << "  keygen=" << settings.keygen_threads
                      << "  flush=" << settings.match_drain_parallel
                      << "  lanes<= " << settings.cuda_max_lanes
                      << "  vram_cap=" << settings.target_vram_pct << "%\n";
            if (prop.persistingL2CacheMaxSize > 0) {
                cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
            }
            if (prop.major >= 8) {
                cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 128);
            }
        }
    }

    if (diagnose) {
        int devices = 0;
        cudaError_t st = cudaGetDeviceCount(&devices);
        std::cout << "{\n"
                  << "  \"app\": \"" << xn::kAppName << "\",\n"
                  << "  \"version\": \"" << xn::kMinerVersion << "\",\n"
                  << "  \"address\": \"" << settings.address << "\",\n"
                  << "  \"worker\": \"" << settings.worker << "\",\n"
                  << "  \"base_url\": \"" << settings.base_url << "\",\n"
                  << "  \"device_id\": " << settings.device_id << ",\n"
                  << "  \"max_lanes\": " << settings.cuda_max_lanes << ",\n"
                  << "  \"cuda_devices\": " << (st == cudaSuccess ? devices : -1) << "\n"
                  << "}\n";
        return 0;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
#ifdef _WIN32
    SetConsoleCtrlHandler(console_handler, TRUE);
#endif

    bool use_dashboard = settings.dashboard_enabled && !no_dashboard;
    xn::Supervisor supervisor(settings, use_dashboard);
    g_supervisor = &supervisor;

    if (!supervisor.startup_checks()) {
        g_supervisor = nullptr;
        return 1;
    }

    std::cout << xn::kAppName << " " << xn::kMinerVersion << "\n";
    supervisor.run(max_seconds);
    const bool want_update = supervisor.update_requested();
    g_supervisor = nullptr;
    return want_update ? xn::kExitCodeUpdate : 0;
}
