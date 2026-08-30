#include "KeygenPool.h"

#include "../RandomHexKeyGenerator.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace hashapi {
namespace {

constexpr int kDefaultThreads = 4;
constexpr int kMinThreads = 2;

struct Job {
    char* dst = nullptr;
    std::size_t count = 0;
    std::size_t key_length = 64;
    const std::string* prefix = nullptr;
    std::atomic<int>* remaining = nullptr;
    std::mutex* done_mu = nullptr;
    std::condition_variable* done_cv = nullptr;
};

class KeygenPool {
public:
    static KeygenPool& instance() {
        static KeygenPool pool;
        return pool;
    }

    void configure(int threads) {
        int logical = 0;
#ifdef _WIN32
        SYSTEM_INFO info{};
        GetSystemInfo(&info);
        logical = static_cast<int>(info.dwNumberOfProcessors);
#else
        const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        logical = static_cast<int>(ncpu > 0 ? ncpu : 2);
#endif
        if (logical < 1) logical = 2;
        const int cap = std::max(kMinThreads, logical);
        const int n = std::max(kMinThreads, std::min(cap, threads > 0 ? threads : kDefaultThreads));
        if (started_ && n == threads_.load()) return;
        stop();
        threads_.store(n);
        start();
    }

    int threads() const { return threads_.load(); }

    void fill(char* arena, std::size_t count, std::size_t key_length, const std::string& prefix) {
        if (arena == nullptr || count == 0 || key_length == 0) return;
        if (!started_) configure(kDefaultThreads);

        const int n = std::max(1, threads_.load());
        const std::size_t use =
            std::max<std::size_t>(1, std::min(static_cast<std::size_t>(n), count));

        std::atomic<int> remaining{static_cast<int>(use)};
        std::mutex done_mu;
        std::condition_variable done_cv;

        {
            std::lock_guard<std::mutex> lock(q_mu_);
            const std::size_t chunk = (count + use - 1) / use;
            for (std::size_t t = 0; t < use; ++t) {
                const std::size_t begin = t * chunk;
                if (begin >= count) {
                    remaining.fetch_sub(1, std::memory_order_relaxed);
                    continue;
                }
                const std::size_t end = std::min(count, begin + chunk);
                Job j;
                j.dst = arena + begin * key_length;
                j.count = end - begin;
                j.key_length = key_length;
                j.prefix = &prefix;
                j.remaining = &remaining;
                j.done_mu = &done_mu;
                j.done_cv = &done_cv;
                q_.push_back(j);
            }
        }
        q_cv_.notify_all();

        std::unique_lock<std::mutex> lock(done_mu);
        done_cv.wait(lock, [&] { return remaining.load(std::memory_order_acquire) <= 0; });
    }

    ~KeygenPool() { stop(); }

private:
    void start() {
        stop_ = false;
        workers_.reserve(static_cast<std::size_t>(threads_));
        for (int i = 0; i < threads_; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
        pin_workers();
        started_ = true;
    }

    void stop() {
        if (!started_) return;
        {
            std::lock_guard<std::mutex> lock(q_mu_);
            stop_ = true;
        }
        q_cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();
        started_ = false;
        stop_ = false;
    }

    void worker_loop() {
        RandomHexKeyGenerator gen("", 64);
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(q_mu_);
                q_cv_.wait(lock, [&] { return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) return;
                if (q_.empty()) continue;
                job = q_.front();
                q_.pop_front();
            }
            if (job.dst != nullptr && job.count > 0 && job.prefix != nullptr) {
                gen.setPrefix(*job.prefix);
                gen.fillMany(job.dst, job.count);
            }
            if (job.remaining != nullptr &&
                job.remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                if (job.done_mu != nullptr && job.done_cv != nullptr) {
                    std::lock_guard<std::mutex> lock(*job.done_mu);
                    job.done_cv->notify_all();
                }
            }
        }
    }

    void pin_workers() {
        // Do not pin to a 6-core CCD. Let the OS schedule across the cores we
        // already sized the pool for — pinning over-expresses small CPUs.
        (void)workers_;
    }

    std::mutex q_mu_;
    std::condition_variable q_cv_;
    std::deque<Job> q_;
    std::vector<std::thread> workers_;
    std::atomic<int> threads_{kDefaultThreads};
    bool started_ = false;
    bool stop_ = false;
};

}  // namespace

void configureKeygenPool(int threads) { KeygenPool::instance().configure(threads); }

int keygenPoolThreads() { return KeygenPool::instance().threads(); }

void keygenFillFlat(char* arena, std::size_t count, std::size_t key_length,
                    const std::string& prefix) {
    KeygenPool::instance().fill(arena, count, key_length, prefix);
}

}  // namespace hashapi
