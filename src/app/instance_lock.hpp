#pragma once

#include <filesystem>
#include <string>

namespace xn {

class InstanceLock {
public:
    explicit InstanceLock(std::filesystem::path path);
    ~InstanceLock();

    bool acquire();
    void release();

private:
    std::filesystem::path path_;
    bool held_ = false;
#ifdef _WIN32
    void* handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace xn
