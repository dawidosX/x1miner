#include "CudaHashBackend.h"

#include "HashApiEncoding.h"
#include "HashApiMatching.h"
#include "HashApiValidation.h"
#include "KeygenPool.h"
#include "../ComputeBackend.h"
#include "../RandomHexKeyGenerator.h"
#include "../argon2-common.h"
#include "../argon2params.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <future>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace hashapi {
namespace {

double elapsedMillis(std::chrono::steady_clock::time_point start,
                     std::chrono::steady_clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

constexpr std::size_t kMinParallelFirstBlockAttempts = 8;
constexpr std::size_t kFinalizeTimingChunkSize = 64;
// Multi-lane mining: each lane finalizes concurrently. Cap workers so 8 lanes do not
// spawn 8 * hardware_concurrency threads (create/join thrash + CPU oversubscription).
constexpr std::size_t kMaxFinalizeWorkers = 2;
// 64-byte digest → argon2-style base64 (no padding) is 86 chars.
constexpr std::size_t kDigestBase64Max = 88;

std::size_t cappedWorkerCount(std::size_t attempts, std::size_t hard_cap, std::size_t min_for_parallel)
{
    if (attempts < min_for_parallel || hard_cap <= 1) {
        return 1;
    }
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const std::size_t hw = hardware_threads == 0 ? hard_cap : static_cast<std::size_t>(hardware_threads);
    // Leave headroom for other lanes / submit worker.
    const std::size_t budget = std::max<std::size_t>(1, hw / 4);
    return std::max<std::size_t>(1, std::min({hard_cap, budget, attempts}));
}

void fillPasswordBlock(ComputeBackend& backend,
                       const Argon2Params& params,
                       std::size_t index,
                       const std::string& password,
                       Argon2FirstBlockTimings* timings)
{
    if (timings != nullptr) {
        params.fillFirstBlocks(backend.getInputMemory(index), password.c_str(), password.size(), timings);
    } else {
        params.fillFirstBlocks(backend.getInputMemory(index), password.c_str(), password.size());
    }
}

std::size_t firstBlockWorkerCount(std::size_t attempts, std::size_t worker_cap)
{
    if (attempts < kMinParallelFirstBlockAttempts) {
        return 1;
    }

    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads < 2) {
        return 1;
    }

    std::size_t worker_count = std::min<std::size_t>(attempts, hardware_threads);
    if (worker_cap > 0) {
        worker_count = std::min(worker_count, worker_cap);
    }
    return std::max<std::size_t>(1, worker_count);
}

std::size_t firstBlockChunkSize(std::size_t attempts, std::size_t worker_count)
{
    if (attempts == 0 || worker_count == 0) {
        return 0;
    }
    return (attempts + worker_count - 1) / worker_count;
}

std::size_t firstBlockSelectedChunkSize(std::size_t attempts,
                                        std::size_t worker_count,
                                        std::size_t dynamic_chunk_size)
{
    if (attempts == 0 || worker_count == 0) {
        return 0;
    }
    if (dynamic_chunk_size > 0 && worker_count > 1) {
        return std::min(attempts, dynamic_chunk_size);
    }
    return firstBlockChunkSize(attempts, worker_count);
}

std::size_t recommendedFirstBlockDynamicChunkSize(const HashApiRequest& request,
                                                  std::size_t attempts,
                                                  std::size_t worker_count)
{
    if (!request.first_block_dynamic_chunk_auto ||
        request.backend != "cuda" ||
        !request.key.empty() ||
        attempts < 1024 ||
        worker_count <= 1) {
        return 0;
    }
    if (request.difficulty == 1) {
        return 16;
    }
    if (request.difficulty == 8) {
        return attempts >= 2048 ? 16 : 32;
    }
    if (request.difficulty == 64) {
        return attempts <= 2048 ? 16 : 0;
    }
    return 0;
}

std::uint8_t decodeHexNibble(char value)
{
    if (value >= '0' && value <= '9') {
        return static_cast<std::uint8_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
        return static_cast<std::uint8_t>(value - 'a' + 10);
    }
    if (value >= 'A' && value <= 'F') {
        return static_cast<std::uint8_t>(value - 'A' + 10);
    }
    throw std::invalid_argument("salt contains non-hex character");
}

std::vector<std::uint8_t> decodeHexBytes(const std::string& hex)
{
    std::vector<std::uint8_t> bytes;
    bytes.reserve(hex.size() / 2);
    for (std::size_t i = 0; i < hex.size(); i += 2) {
        const std::uint8_t high = decodeHexNibble(hex[i]);
        const std::uint8_t low = decodeHexNibble(hex[i + 1]);
        bytes.push_back(static_cast<std::uint8_t>((high << 4) | low));
    }
    return bytes;
}

void fillPasswordBlocks(ComputeBackend& backend,
                        const Argon2Params& params,
                        const std::vector<std::string>& passwords,
                        std::size_t worker_cap,
                        std::size_t dynamic_chunk_size,
                        Argon2FirstBlockTimings* timings)
{
    const std::size_t attempts = passwords.size();
    const std::size_t worker_count = firstBlockWorkerCount(attempts, worker_cap);
    if (worker_count <= 1) {
        for (std::size_t i = 0; i < attempts; ++i) {
            fillPasswordBlock(backend, params, i, passwords[i], timings);
        }
        return;
    }

    const std::size_t chunk_size = firstBlockSelectedChunkSize(attempts, worker_count, dynamic_chunk_size);
    std::vector<Argon2FirstBlockTimings> worker_timings(timings == nullptr ? 0 : worker_count);
    std::vector<std::thread> workers;
    std::vector<double> worker_start_offsets(timings == nullptr ? 0 : worker_count);
    std::vector<double> worker_finish_offsets(timings == nullptr ? 0 : worker_count);
    std::atomic<std::size_t> next_dynamic_index{0};
    const auto launch_start = timings == nullptr
        ? std::chrono::steady_clock::time_point{}
        : std::chrono::steady_clock::now();
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const bool dynamic_chunks = dynamic_chunk_size > 0;
        const std::size_t static_begin = worker * chunk_size;
        const std::size_t static_end = std::min(attempts, static_begin + chunk_size);
        if (!dynamic_chunks && static_begin >= static_end) {
            break;
        }
        workers.emplace_back([&backend,
                              &params,
                              &passwords,
                              &worker_timings,
                              &worker_start_offsets,
                              &worker_finish_offsets,
                              &next_dynamic_index,
                              dynamic_chunks,
                              timings,
                              worker,
                              static_begin,
                              static_end,
                              attempts,
                              chunk_size,
                              launch_start]() {
            std::chrono::steady_clock::time_point worker_start;
            if (timings != nullptr) {
                worker_start = std::chrono::steady_clock::now();
                worker_start_offsets[worker] = elapsedMillis(launch_start, worker_start);
            }
            Argon2FirstBlockTimings* local_timings = timings == nullptr ? nullptr : &worker_timings[worker];
            if (dynamic_chunks) {
                for (;;) {
                    const std::size_t begin = next_dynamic_index.fetch_add(chunk_size, std::memory_order_relaxed);
                    if (begin >= attempts) {
                        break;
                    }
                    const std::size_t end = std::min(attempts, begin + chunk_size);
                    for (std::size_t i = begin; i < end; ++i) {
                        fillPasswordBlock(backend, params, i, passwords[i], local_timings);
                    }
                }
            } else {
                for (std::size_t i = static_begin; i < static_end; ++i) {
                    fillPasswordBlock(backend, params, i, passwords[i], local_timings);
                }
            }
            if (local_timings != nullptr) {
                const auto worker_finish = std::chrono::steady_clock::now();
                local_timings->worker_ms = elapsedMillis(worker_start, worker_finish);
                worker_finish_offsets[worker] = elapsedMillis(launch_start, worker_finish);
            }
        });
    }
    if (timings != nullptr) {
        timings->thread_launch_ms = elapsedMillis(launch_start, std::chrono::steady_clock::now());
    }

    for (std::thread& worker : workers) {
        worker.join();
    }
    if (timings != nullptr) {
        double min_worker_start_ms = 0.0;
        for (const Argon2FirstBlockTimings& item : worker_timings) {
            timings->initial_hash_ms += item.initial_hash_ms;
            timings->digest_ms += item.digest_ms;
            timings->worker_ms = std::max(timings->worker_ms, item.worker_ms);
        }
        for (std::size_t worker = 0; worker < workers.size(); ++worker) {
            const double worker_start_ms = worker_start_offsets[worker];
            const double worker_finish_ms = worker_finish_offsets[worker];
            if (worker == 0 || worker_start_ms < min_worker_start_ms) {
                min_worker_start_ms = worker_start_ms;
            }
            timings->max_worker_start_ms = std::max(timings->max_worker_start_ms, worker_start_ms);
            timings->max_worker_finish_ms = std::max(timings->max_worker_finish_ms, worker_finish_ms);
        }
        timings->worker_start_span_ms = timings->max_worker_start_ms - min_worker_start_ms;
        timings->worker_finish_span_ms = timings->max_worker_finish_ms - min_worker_start_ms;
    }
}

} // namespace

CudaHashBackend::CudaHashBackend(ComputeBackend& backend)
    : backend_(&backend)
{
}

CudaHashBackend::CudaHashBackend(std::unique_ptr<ComputeBackend> backend)
    : backend_(backend.get()), owned_backend_(std::move(backend))
{
    if (backend_ == nullptr) {
        throw std::invalid_argument("cuda backend cannot be null");
    }
}

CudaHashBackend::~CudaHashBackend()
{
    if (next_keygen_.valid()) {
        next_keygen_.wait();
    }
    // Drop any buffered GPU results (process exit / lane teardown). Prefer drainPipeline() first.
    pending_ = {};
}

ComputeBackend& CudaHashBackend::backend()
{
    if (backend_ == nullptr) {
        throw std::runtime_error("cuda backend is not initialized");
    }
    return *backend_;
}

const ComputeBackend& CudaHashBackend::backend() const
{
    if (backend_ == nullptr) {
        throw std::runtime_error("cuda backend is not initialized");
    }
    return *backend_;
}

void CudaHashBackend::finalizeSlot(const PipelinedBatch& slot, HashApiResult& result) const
{
    if (!slot.ready || slot.attempts == 0) {
        result.ok = true;
        return;
    }

    HashApiRequest match_req;
    match_req.target_pattern = slot.target_pattern;
    match_req.allow_xuni = slot.allow_xuni;

    // Prefer 1–2 workers: multi-lane already parallelizes across lanes.
    const std::size_t nthreads = cappedWorkerCount(slot.attempts, kMaxFinalizeWorkers, 4096);

    std::vector<std::vector<HashApiMatch>> thread_matches(nthreads);
    std::vector<std::string> single_key_hash(nthreads);
    const std::size_t key_len = slot.key_length > 0 ? slot.key_length : kHashApiKeyLength;
    constexpr std::size_t kDigest = kDefaultHashLength;
    constexpr std::size_t kBlock = static_cast<std::size_t>(argon2::ARGON2_BLOCK_SIZE);

    auto worker = [&](std::size_t tid, std::size_t begin, std::size_t end) {
        std::array<std::uint8_t, kDefaultHashLength> digest{};
        char b64[kDigestBase64Max];
        HashApiResult local;
        local.matches.reserve(4);
        for (std::size_t i = begin; i < end; ++i) {
            const std::uint8_t* digest_ptr = nullptr;
            if (slot.gpu_finalized) {
                digest_ptr = slot.digests.data() + i * kDigest;
            } else {
                const void* mem = slot.digests.data() + i * kBlock;
                slot.params.finalize(digest.data(), mem);
                digest_ptr = digest.data();
            }
            const std::size_t b64_len =
                base64EncodeTo(b64, sizeof(b64), digest_ptr, kDefaultHashLength);
            if (b64_len == 0) {
                continue;
            }
            if (slot.single_key) {
                single_key_hash[tid].assign(b64, b64_len);
            }
            const char* key_ptr = reinterpret_cast<const char*>(slot.keys.data() + i * key_len);
            // No per-attempt std::string until a pattern actually matches.
            appendMatchesRaw(match_req, local, key_ptr, key_len, b64, b64_len, i);
        }
        thread_matches[tid] = std::move(local.matches);
    };

    if (nthreads <= 1) {
        worker(0, 0, slot.attempts);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        const std::size_t chunk = (slot.attempts + nthreads - 1) / nthreads;
        for (std::size_t t = 0; t < nthreads; ++t) {
            const std::size_t begin = t * chunk;
            if (begin >= slot.attempts) break;
            const std::size_t end = std::min(slot.attempts, begin + chunk);
            pool.emplace_back(worker, t, begin, end);
        }
        for (auto& th : pool) th.join();
    }

    result.matches.clear();
    for (auto& tm : thread_matches) {
        result.matches.insert(result.matches.end(),
                              std::make_move_iterator(tm.begin()),
                              std::make_move_iterator(tm.end()));
    }
    if (slot.single_key) {
        for (auto& h : single_key_hash) {
            if (!h.empty()) {
                result.hash = std::move(h);
                break;
            }
        }
    }
    result.ok = true;
    result.attempts = slot.attempts;
    result.batch_size = slot.attempts;
    result.batch_size_min = slot.attempts;
    result.batch_size_max = slot.attempts;
    result.request_id = slot.request_id;
}

void CudaHashBackend::freezeHashedKeys(const std::uint8_t* src, std::size_t attempts,
                                       std::size_t key_len) {
    hashed_attempts_ = 0;
    hashed_key_len_ = key_len;
    hashed_keys_.clear();
    if (src == nullptr || attempts == 0 || key_len == 0) return;
    const std::size_t n = attempts * key_len;
    hashed_keys_.assign(src, src + n);
    hashed_attempts_ = attempts;
    // Bind once per hash wave. A second freeze before collect used to pair
    // the next-slot keys with this wave's digests (same XEN11, wrong key).
    if (!bound_ready_) {
        bound_keys_ = hashed_keys_;
        bound_attempts_ = attempts;
        bound_ready_ = true;
    }
}

PipelinedBatch CudaHashBackend::snapshotOutputs(const HashApiRequest& request,
                                                Argon2Params params,
                                                const std::uint8_t* keys,
                                                std::size_t key_length,
                                                bool single_key,
                                                std::size_t attempts) const
{
    PipelinedBatch slot;
    slot.ready = true;
    slot.params = std::move(params);
    slot.target_pattern = request.target_pattern;
    slot.allow_xuni = request.allow_xuni;
    slot.single_key = single_key;
    slot.request_id = request.request_id;
    slot.attempts = attempts;
    slot.difficulty = request.difficulty;
    slot.key_length = key_length;

    const std::size_t key_bytes = attempts * key_length;
    if (keys == nullptr || key_bytes == 0 || attempts == 0) {
        slot.ready = false;
        return slot;
    }
    // First key in the live slot must be 64 hex. A zeroed/wrong-slot snapshot
    // would otherwise attach XEN11 digests to NUL keys and fail /verify.
    if (key_length == 64) {
        bool hex = true;
        for (std::size_t i = 0; i < 64; ++i) {
            const unsigned char c = keys[i];
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
                hex = false;
                break;
            }
        }
        if (!hex) {
            slot.ready = false;
            return slot;
        }
    }
    slot.keys.resize(key_bytes);
    std::memcpy(slot.keys.data(), keys, key_bytes);

    auto& compute = backend();
    slot.gpu_finalized = compute.hasDeviceFinalHashes();
    if (slot.gpu_finalized) {
        constexpr std::size_t kDigest = kDefaultHashLength;
        const void* base = compute.getFinalHashMemory(0);
        if (base == nullptr) {
            slot.gpu_finalized = false;
        } else {
            // Contiguous pinned host digests — one bulk copy, not N small ones.
            const std::size_t bytes = attempts * kDigest;
            slot.digests.resize(bytes);
            std::memcpy(slot.digests.data(), base, bytes);
        }
    }
    if (!slot.gpu_finalized) {
        // Capability fallback: last 1024-byte Argon2 block → CPU Blake2b-long.
        constexpr std::size_t kBlock = static_cast<std::size_t>(argon2::ARGON2_BLOCK_SIZE);
        slot.digests.resize(attempts * kBlock);
        for (std::size_t i = 0; i < attempts; ++i) {
            std::memcpy(slot.digests.data() + i * kBlock,
                        compute.getOutputMemory(i),
                        kBlock);
        }
    }
    return slot;
}

HashApiResult CudaHashBackend::drainPipeline()
{
    HashApiResult result;
    result.backend = "cuda";
    result.ok = true;
    if (!pending_.ready) {
        return result;
    }
    finalizeSlot(pending_, result);
    pending_ = {};
    return result;
}

void CudaHashBackend::ensureInitialized(ComputeBackend& backend,
                                        const Argon2Params& params,
                                        std::size_t batch_size)
{
    const auto segment_blocks = params.getSegmentBlocks();
    if (initialized_ &&
        initialized_batch_size_ == batch_size &&
        initialized_segment_blocks_ == segment_blocks) {
        return;
    }

    // Serialize first-time cudaMalloc across lanes. Concurrent 8x~3 GiB
    // allocations on WDDM have aborted the process on some desktops.
    static std::mutex init_mu;
    std::lock_guard<std::mutex> init_lock(init_mu);
    if (initialized_ &&
        initialized_batch_size_ == batch_size &&
        initialized_segment_blocks_ == segment_blocks) {
        return;
    }

    backend.init(batch_size, argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                 1, 1, segment_blocks);
    initialized_ = true;
    initialized_batch_size_ = batch_size;
    initialized_segment_blocks_ = segment_blocks;
}

HashApiResult CudaHashBackend::runBatch(const HashApiRequest& request)
{
    const auto total_start = std::chrono::steady_clock::now();
    HashApiResult result;
    result.request_id = request.request_id;
    result.algorithm = request.algorithm;
    result.backend = "cuda";
    result.device_id = request.device_id;
    result.batch_size = request.batch_size;
    result.gpu_first_blocks = request.gpu_first_blocks;

    const auto validation_start = std::chrono::steady_clock::now();
    const auto errors = validateRequest(request);
    result.timings.validation_ms = elapsedMillis(validation_start, std::chrono::steady_clock::now());
    if (!errors.empty()) {
        result.error = joinErrors(errors);
        result.timings.total_ms = elapsedMillis(total_start, std::chrono::steady_clock::now());
        return result;
    }
    if (request.backend != "cuda") {
        result.error = "CudaHashBackend requires backend=cuda";
        result.timings.total_ms = elapsedMillis(total_start, std::chrono::steady_clock::now());
        return result;
    }

    const auto start = std::chrono::steady_clock::now();

    try {
        const auto setup_start = std::chrono::steady_clock::now();
        auto timed_setup_step = [&request](auto&& action) {
            if (!request.detailed_timings) {
                action();
                return 0.0;
            }
            const auto step_start = std::chrono::steady_clock::now();
            action();
            return elapsedMillis(step_start, std::chrono::steady_clock::now());
        };
        std::string salt;
        std::string prefix;
        std::string fixed_key;
        result.timings.setup_normalize_cpu_ms = timed_setup_step([&]() {
            salt = normalizeHex(request.salt_hex);
            prefix = normalizeHex(request.key_prefix);
            fixed_key = normalizeHex(request.key);
        });
        const bool single_key = !fixed_key.empty();
        const std::size_t attempts = single_key ? 1 : request.batch_size;
        result.first_block_worker_count = firstBlockWorkerCount(attempts, request.first_block_workers);
        result.first_block_dynamic_chunk_size = 0;
        result.first_block_dynamic_chunk_auto =
            request.first_block_dynamic_chunk_auto && request.first_block_dynamic_chunk_size == 0;
        const std::size_t requested_dynamic_chunk_size = request.first_block_dynamic_chunk_size > 0
            ? request.first_block_dynamic_chunk_size
            : recommendedFirstBlockDynamicChunkSize(request, attempts, result.first_block_worker_count);
        if (result.first_block_worker_count > 1 && requested_dynamic_chunk_size > 0) {
            result.first_block_dynamic_chunk_size = std::min(attempts, requested_dynamic_chunk_size);
        }
        result.first_block_chunk_size = firstBlockSelectedChunkSize(
            attempts,
            result.first_block_worker_count,
            result.first_block_dynamic_chunk_size);
        result.first_block_dynamic_chunk_size_min = result.first_block_dynamic_chunk_size;
        result.first_block_dynamic_chunk_size_max = result.first_block_dynamic_chunk_size;
        result.first_block_chunk_size_min = result.first_block_chunk_size;
        result.first_block_chunk_size_max = result.first_block_chunk_size;

        auto& compute_backend = backend();
        result.timings.setup_activate_cpu_ms = timed_setup_step([&]() {
            compute_backend.activate();
        });
        if (!device_info_cached_) {
            result.timings.setup_device_info_cpu_ms = timed_setup_step([&]() {
                const DeviceInfo device_info = compute_backend.getDeviceInfo();
                cached_device_id_ = device_info.index;
                device_info_cached_ = true;
            });
        }
        result.device_id = device_info_cached_ ? cached_device_id_ : request.device_id;

        Argon2Params params;
        result.timings.setup_params_cpu_ms = timed_setup_step([&]() {
            params = Argon2Params(argon2::ARGON2_ID, argon2::ARGON2_VERSION_13,
                                  kDefaultHashLength, salt, nullptr, 0, nullptr, 0,
                                  1, request.difficulty, 1);
        });
        result.timings.setup_backend_init_cpu_ms = timed_setup_step([&]() {
            // Champ pair-queue (wave_role 1/2/3) keeps dual patches. Windows
            // Desktop full-wave (role 0) uses a single arena like xnminer.exe.
            if (request.wave_role != 0) {
                int patches = request.work_patches;
                if (patches < 2) patches = 2;
                if (patches > 3) patches = 3;
                compute_backend.setWorkPatches(patches);
                std::size_t arena = request.arena_batch_size;
                if (arena < attempts) arena = attempts;
                ensureInitialized(compute_backend, params, arena);
                if (attempts < arena) {
                    ensureInitialized(compute_backend, params, attempts);
                }
            } else {
                ensureInitialized(compute_backend, params, attempts);
            }
        });
        result.timings.setup_ms = elapsedMillis(setup_start, std::chrono::steady_clock::now());

        const std::size_t key_len = kHashApiKeyLength;
        const std::size_t key_bytes = attempts * key_len;
        const int wave_role = request.wave_role;

        if (wave_role == 1 && request.gpu_first_blocks && !single_key) {
            // Create: first-block. Keys should already be staged during the last launch.
            compute_backend.setHashPatch(0);
            int slot = 0;
            if (preload_ready_ && preload_attempts_ == attempts) {
                slot = preload_slot_;
                preload_ready_ = false;
            } else {
                uint8_t* arena = compute_backend.ensureHostKeyBufferSlot(0, key_bytes);
                if (arena == nullptr) throw std::runtime_error("cuda backend host key buffer unavailable");
                keygenFillFlat(reinterpret_cast<char*>(arena), attempts, key_len, prefix);
                preload_ready_ = false;
            }
            live_slot_ = slot;
            compute_backend.setLiveKeySlot(slot);
            uint8_t* arena = compute_backend.ensureHostKeyBufferSlot(slot, key_bytes);
            if (arena == nullptr) throw std::runtime_error("cuda backend host key buffer unavailable");
            if (cached_salt_hex_ != salt) {
                cached_salt_hex_ = salt;
                cached_salt_bytes_ = decodeHexBytes(salt);
            }
            if (!compute_backend.prepareInputBlocksOnDeviceFlat(
                    key_len, cached_salt_bytes_, params.getOutputLength(), params.getMemoryCost(),
                    params.getTimeCost(), params.getVersion(), params.getType(),
                    params.getLanes())) {
                throw std::runtime_error("cuda backend does not support gpu_first_blocks");
            }
            compute_backend.runPrepareOnly();
            compute_backend.finish();
            freezeHashedKeys(arena, attempts, key_len);
            result.ok = true;
            result.attempts = 0;
            result.batch_size = attempts;
            result.batch_size_min = attempts;
            result.batch_size_max = attempts;
        } else if (wave_role == 3 && request.gpu_first_blocks) {
            // Freeze live keys BEFORE filling the next slot (champ pairing).
            uint8_t* live = compute_backend.ensureHostKeyBufferSlot(live_slot_, key_bytes);
            freezeHashedKeys(live, attempts, key_len);
            compute_backend.setHashPatch(0);
            compute_backend.runHashOnly();
            if (!single_key) {
                const int next = (live_slot_ == 0) ? 1 : 0;
                uint8_t* next_arena = compute_backend.ensureHostKeyBufferSlot(next, key_bytes);
                if (next_arena != nullptr) {
                    keygenFillFlat(reinterpret_cast<char*>(next_arena), attempts, key_len, prefix);
                    preload_ready_ = true;
                    preload_slot_ = next;
                    preload_attempts_ = attempts;
                }
            }
            result.ok = true;
            result.attempts = 0;
            result.batch_size = attempts;
            result.batch_size_min = attempts;
            result.batch_size_max = attempts;
        } else if (wave_role == 2 && request.gpu_first_blocks) {
            // Collect: digests + keys frozen at launch — never the next-key slot.
            result.timings.kernel_ms = static_cast<double>(compute_backend.finish());
            const uint8_t* keys = nullptr;
            if (bound_ready_ && bound_attempts_ == attempts && bound_keys_.size() == key_bytes) {
                keys = bound_keys_.data();
            } else if (hashed_attempts_ == attempts && hashed_keys_.size() == key_bytes) {
                keys = hashed_keys_.data();
            }
            PipelinedBatch current =
                snapshotOutputs(request, params, keys, key_len, false, attempts);
            bound_ready_ = false;
            bound_keys_.clear();
            bound_attempts_ = 0;
            if (pending_.ready) {
                HashApiResult prev;
                prev.backend = "cuda";
                finalizeSlot(pending_, prev);
                result.matches = std::move(prev.matches);
                if (!prev.hash.empty()) result.hash = std::move(prev.hash);
            }
            pending_ = std::move(current);
            result.ok = true;
            result.attempts = attempts;
            result.batch_size = attempts;
            result.batch_size_min = attempts;
            result.batch_size_max = attempts;
        } else {
        // Windows Desktop full wave: async-finalize the previous snapshot so it
        // overlaps this lane's keygen + GPU. Do NOT keygen the next patch
        // before finish() — that left the GPU under-utilized.

        std::future<HashApiResult> prev_finalize;
        if (pending_.ready) {
            PipelinedBatch prev_slot = std::move(pending_);
            pending_ = {};
            prev_finalize = std::async(std::launch::async, [this, slot = std::move(prev_slot)]() mutable {
                HashApiResult fr;
                fr.backend = "cuda";
                finalizeSlot(slot, fr);
                return fr;
            });
        }

        const auto input_start = std::chrono::steady_clock::now();
        Argon2FirstBlockTimings first_block_timings;
        Argon2FirstBlockTimings* detailed_first_block_timings =
            request.detailed_timings ? &first_block_timings : nullptr;

        // Persistent pinned key arena (grows once to max batch, never frees strings per job).
        const std::size_t key_len = kHashApiKeyLength;
        const std::size_t key_bytes = attempts * key_len;
        uint8_t* key_arena = compute_backend.ensureHostKeyBuffer(key_bytes);
        if (key_arena == nullptr) {
            throw std::runtime_error("cuda backend host key buffer unavailable");
        }

        if (request.gpu_first_blocks) {
            const auto keygen_start = std::chrono::steady_clock::now();
            if (next_keygen_.valid()) {
                next_keygen_.wait();
                next_keygen_ = {};
            }
            if (single_key) {
                if (fixed_key.size() != key_len) {
                    throw std::runtime_error("fixed key length mismatch");
                }
                std::memcpy(key_arena, fixed_key.data(), key_len);
            } else if (next_keys_.size() == key_bytes) {
                std::memcpy(key_arena, next_keys_.data(), key_bytes);
            } else {
                keygenFillFlat(reinterpret_cast<char*>(key_arena), attempts, key_len, prefix);
            }
            preload_ready_ = false;
            result.timings.keygen_ms = elapsedMillis(keygen_start, std::chrono::steady_clock::now());

            const auto device_first_block_start = std::chrono::steady_clock::now();
            if (cached_salt_hex_ != salt) {
                cached_salt_hex_ = salt;
                cached_salt_bytes_ = decodeHexBytes(salt);
            }
            if (!compute_backend.prepareInputBlocksOnDeviceFlat(key_len,
                                                                cached_salt_bytes_,
                                                                params.getOutputLength(),
                                                                params.getMemoryCost(),
                                                                params.getTimeCost(),
                                                                params.getVersion(),
                                                                params.getType(),
                                                                params.getLanes())) {
                throw std::runtime_error("cuda backend does not support gpu_first_blocks");
            }
            result.timings.first_block_ms += elapsedMillis(device_first_block_start,
                                                           std::chrono::steady_clock::now());
        } else {
            // CPU first-block path (fallback): still use flat keys for match reporting.
            std::vector<std::string> password_storage;
            password_storage.reserve(attempts);
            if (result.first_block_worker_count <= 1) {
                if (single_key) {
                    const auto keygen_start = std::chrono::steady_clock::now();
                    password_storage.push_back(fixed_key);
                    std::memcpy(key_arena, fixed_key.data(),
                                std::min(key_len, fixed_key.size()));
                    result.timings.keygen_ms +=
                        elapsedMillis(keygen_start, std::chrono::steady_clock::now());
                    fillPasswordBlock(compute_backend, params, 0, password_storage.front(),
                                      detailed_first_block_timings);
                } else {
                    RandomHexKeyGenerator key_generator(prefix, key_len);
                    for (std::size_t i = 0; i < attempts; ++i) {
                        const std::string key = key_generator.nextRandomKey();
                        password_storage.push_back(key);
                        std::memcpy(key_arena + i * key_len, key.data(), key_len);
                        fillPasswordBlock(compute_backend, params, i, key,
                                          detailed_first_block_timings);
                    }
                }
            } else {
                RandomHexKeyGenerator key_generator(prefix, key_len);
                for (std::size_t i = 0; i < attempts; ++i) {
                    password_storage.push_back(key_generator.nextRandomKey());
                    std::memcpy(key_arena + i * key_len, password_storage.back().data(), key_len);
                }
                fillPasswordBlocks(compute_backend, params, password_storage,
                                   request.first_block_workers,
                                   result.first_block_dynamic_chunk_size,
                                   detailed_first_block_timings);
            }
            result.timings.first_block_ms += first_block_timings.worker_ms;
        }
        result.timings.first_block_initial_hash_cpu_ms = first_block_timings.initial_hash_ms;
        result.timings.first_block_digest_cpu_ms = first_block_timings.digest_ms;
        result.timings.first_block_max_worker_ms = first_block_timings.worker_ms;
        result.timings.first_block_thread_launch_ms = first_block_timings.thread_launch_ms;
        result.timings.first_block_max_worker_start_ms = first_block_timings.max_worker_start_ms;
        result.timings.first_block_worker_start_span_ms = first_block_timings.worker_start_span_ms;
        result.timings.first_block_max_worker_finish_ms = first_block_timings.max_worker_finish_ms;
        result.timings.first_block_worker_finish_span_ms = first_block_timings.worker_finish_span_ms;
        result.timings.input_ms = elapsedMillis(input_start, std::chrono::steady_clock::now());

        const auto compute_start = std::chrono::steady_clock::now();
        compute_backend.run();
        // Fill the NEXT key bag while this oneshot runs so the GPU does not
        // sit idle on the following wave (Vast was ~50% util without this).
        if (request.gpu_first_blocks && !single_key && attempts > 1) {
            next_keys_.resize(key_bytes);
            char* dst = reinterpret_cast<char*>(next_keys_.data());
            const std::string pref = prefix;
            const std::size_t n = attempts;
            const std::size_t kl = key_len;
            next_keygen_ = std::async(std::launch::async, [dst, n, kl, pref]() {
                keygenFillFlat(dst, n, kl, pref);
            });
        }
        result.timings.kernel_ms = static_cast<double>(compute_backend.finish());
        result.timings.host_to_device_ms = static_cast<double>(compute_backend.getLastHostToDeviceMs());
        result.timings.gpu_first_block_ms = static_cast<double>(compute_backend.getLastGpuFirstBlockMs());
        result.timings.device_to_host_ms = static_cast<double>(compute_backend.getLastDeviceToHostMs());
        result.timings.compute_ms = elapsedMillis(compute_start, std::chrono::steady_clock::now());

        PipelinedBatch current =
            snapshotOutputs(request, params, key_arena, key_len, single_key, attempts);

        const auto finalize_start = std::chrono::steady_clock::now();
        if (prev_finalize.valid()) {
            HashApiResult prev = prev_finalize.get();
            result.matches = std::move(prev.matches);
            if (!prev.hash.empty()) result.hash = std::move(prev.hash);
        }
        pending_ = std::move(current);
        result.timings.finalize_ms = elapsedMillis(finalize_start, std::chrono::steady_clock::now());
        result.timings.finalize_hash_ms = result.timings.finalize_ms;
        result.timings.argon2_finalize_ms = result.timings.finalize_ms;
        result.timings.base64_ms = 0.0;
        result.timings.match_ms = 0.0;

        result.ok = true;
        result.attempts = attempts;
        result.batch_size = attempts;
        result.batch_size_min = attempts;
        result.batch_size_max = attempts;
        }  // full wave
    } catch (const std::exception& ex) {
        result.error = ex.what();
        // On failure, drop pending so we do not finalise corrupt state later.
        pending_ = {};
    }

    const auto end = std::chrono::steady_clock::now();
    result.elapsed_ms = elapsedMillis(start, end);
    result.timings.total_ms = elapsedMillis(total_start, end);
    if (result.elapsed_ms > 0.0 && result.attempts > 0) {
        result.hashrate = static_cast<double>(result.attempts) / (result.elapsed_ms / 1000.0);
    }

    return result;
}

} // namespace hashapi
