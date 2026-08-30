#include "app/instance_lock.hpp"

#include "util/paths.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#include <fstream>
#include <sstream>

namespace xn {

InstanceLock::InstanceLock(std::filesystem::path path) : path_(std::move(path)) {}

InstanceLock::~InstanceLock() { release(); }

bool InstanceLock::acquire() {
    ensure_parent_dir(path_);
#ifdef _WIN32
    handle_ = CreateFileW(path_.wstring().c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) {
        handle_ = nullptr;
        return false;
    }
    held_ = true;
    return true;
#else
    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) return false;
    if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    if (::ftruncate(fd_, 0) == 0) {
        std::ostringstream oss;
        oss << "pid=" << ::getpid() << "\n";
        const auto s = oss.str();
        (void)::write(fd_, s.data(), s.size());
    }
    held_ = true;
    return true;
#endif
}

void InstanceLock::release() {
    if (!held_) return;
#ifdef _WIN32
    if (handle_) {
        CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
    }
#else
    if (fd_ >= 0) {
        ::flock(fd_, LOCK_UN);
        ::close(fd_);
        fd_ = -1;
    }
#endif
    std::error_code ec;
    std::filesystem::remove(path_, ec);
    held_ = false;
}

}  // namespace xn
