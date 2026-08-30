#include "util/logger.hpp"

#include "util/paths.hpp"

#include <chrono>
#include <fstream>
#include <iostream>

namespace xn {

SessionLogger::SessionLogger(std::filesystem::path path, bool echo_console)
    : echo_console(echo_console), path_(std::move(path)) {
    ensure_parent_dir(path_);
}

void SessionLogger::info(const std::string& msg) { write("INFO", msg); }
void SessionLogger::warn(const std::string& msg) { write("WARN", msg); }
void SessionLogger::error(const std::string& msg) { write("ERROR", msg); }

void SessionLogger::write(const std::string& level, const std::string& msg) {
    std::lock_guard<std::mutex> lock(mu_);
    auto line = now_iso_local() + " [" + level + "] " + msg;
    {
        std::ofstream out(path_, std::ios::app);
        out << line << "\n";
    }
    if (echo_console) {
        if (level == "ERROR") {
            std::cerr << line << std::endl;
        } else {
            std::cout << line << std::endl;
        }
    }
}

}  // namespace xn
