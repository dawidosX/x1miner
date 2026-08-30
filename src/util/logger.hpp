#pragma once

#include <filesystem>
#include <mutex>
#include <string>

namespace xn {

class SessionLogger {
public:
    explicit SessionLogger(std::filesystem::path path, bool echo_console = true);
    void info(const std::string& msg);
    void warn(const std::string& msg);
    void error(const std::string& msg);
    bool echo_console = true;

private:
    void write(const std::string& level, const std::string& msg);
    std::filesystem::path path_;
    std::mutex mu_;
};

}  // namespace xn
