#include "mining/cuda_engine.hpp"

#include "CudaBackend.h"
#include "efficiency/thermal_policy.hpp"
#include "hashapi/CudaHashBackend.h"
#include "hashapi/HashApiTypes.h"
#include "hashapi/KeygenPool.h"
#include "mining/argon2_encode.hpp"
#include "mining/block_types.hpp"
#include "queue/policy.hpp"
#include "util/paths.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace xn {

struct CudaEngine::Lane {
    std::unique_ptr<hashapi::CudaHashBackend> hash;
    std::mutex mu;
};

namespace {

std::string lane_prefix(int lane) {
    std::ostringstream oss;
    oss << std::hex << std::setw(4) << std::setfill('0') << lane;
    return oss.str();
}

}  // namespace

CudaEngine::CudaEngine(const Settings& settings)
    : settings_(settings),
      difficulty_(settings.memory_cost),
      max_lanes_cap_(std::max(1, settings.cuda_max_lanes)),
      config_max_lanes_(std::max(1, settings.cuda_max_lanes)) {}

CudaEngine::~CudaEngine() { stop(); }

void CudaEngine::set_vram_caps(const VramCaps& caps) { vram_caps_ = caps; }

void CudaEngine::set_queue_counts(int pending_xuni, int pending_xnm, int pending_xblk) {
    pending_xuni_ = std::max(0, pending_xuni);
    pending_xnm_ = std::max(0, pending_xnm);
    pending_xblk_ = std::max(0, pending_xblk);
}

bool CudaEngine::compute_allow_xuni_base() {
    // Priority: always hunt XEN11 (XNM preferred over natural XBLK). XUNI only inside
    // the mine window (:56–:04). Taper minute (:04) sparsifies XUNI further so submits win.
    if (!settings_.xuni_mining_enabled) return false;
    if (!in_xuni_window()) {
        xuni_queue_paused_ = false;
        return false;
    }
    const int soft = std::max(0, settings_.xuni_queue_soft_cap);
    const int resume = std::max(0, settings_.xuni_queue_resume);
    if (pending_xuni_ >= soft) {
        xuni_queue_paused_ = true;
    } else if (xuni_queue_paused_ && pending_xuni_ <= resume) {
        xuni_queue_paused_ = false;
    }
    if (xuni_queue_paused_) return false;
    if (settings_.xuni_max_lanes <= 0) return false;

    ++batch_seq_;
    int every = std::max(1, settings_.xuni_every_n_batches);
    // Last minute of the window: slope off XUNI mining (more batches stay XEN11-only).
    if (in_xuni_taper_window()) every = std::max(every * 3, 6);
    if ((batch_seq_ % every) != 0) return false;
    return true;
}

uint64_t CudaEngine::free_vram_bytes() const {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) return static_cast<uint64_t>(free_b);
    return 0;
}

void CudaEngine::sync_lanes() {
    while (static_cast<int>(lanes_impl_.size()) < lanes_) {
        auto lane = std::make_unique<Lane>();
        auto backend = std::make_unique<CudaBackend>(settings_.device_id);
        lane->hash = std::make_unique<hashapi::CudaHashBackend>(std::move(backend));
        lanes_impl_.push_back(std::move(lane));
    }
    while (static_cast<int>(lanes_impl_.size()) > lanes_) {
        lanes_impl_.pop_back();
    }
}

void CudaEngine::stop_lane_workers() {
    {
        std::lock_guard<std::mutex> lock(worker_mu_);
        workers_stop_ = true;
    }
    worker_cv_.notify_all();
    worker_done_cv_.notify_all();
    for (auto& t : lane_workers_) {
        if (t.joinable()) t.join();
    }
    lane_workers_.clear();
    worker_target_lanes_ = 0;
    workers_stop_ = false;
}

void CudaEngine::lane_worker_loop(int lane_index) {
    int seen = 0;
    for (;;) {
        std::unique_lock<std::mutex> lock(worker_mu_);
        worker_cv_.wait(lock, [&] {
            return workers_stop_.load(std::memory_order_relaxed) ||
                   mailboxes_[static_cast<size_t>(lane_index)].seq.load(std::memory_order_relaxed) !=
                       seen;
        });
        if (workers_stop_.load(std::memory_order_relaxed)) {
            return;
        }
        auto& box = mailboxes_[static_cast<size_t>(lane_index)];
        seen = box.seq.load(std::memory_order_relaxed);
        hashapi::HashApiRequest req;
        req.backend = "cuda";
        req.salt_hex = box.salt;
        req.key_prefix = lane_prefix(lane_index);
        req.target_pattern = "XEN11";
        req.difficulty = box.difficulty;
        req.batch_size = box.batch_size;
        req.arena_batch_size = box.arena_batch_size;
        req.allow_xuni = box.allow_xuni_base && (lane_index < box.xuni_lanes);
        req.gpu_first_blocks = true;
        req.first_block_dynamic_chunk_auto = false;
        req.device_id = box.device_id;
        req.work_patches = 0;
        req.wave_role = 0;
        lock.unlock();

        if (req.batch_size == 0 || req.difficulty == 0 || req.salt_hex.empty()) {
            continue;
        }

        if (lane_index >= static_cast<int>(lanes_impl_.size())) {
            box.done_seq.store(seen, std::memory_order_release);
            worker_done_cv_.notify_all();
            continue;
        }

        // Keep hashing this proto until mine_batch publishes a new one.
        // Two-deep ring: GPU starts the next wave while the supervisor consumes.
        while (!workers_stop_.load(std::memory_order_relaxed) &&
               box.seq.load(std::memory_order_relaxed) == seen) {
            {
                std::unique_lock<std::mutex> lk(worker_mu_);
                worker_done_cv_.wait(lk, [&] {
                    return workers_stop_.load(std::memory_order_relaxed) ||
                           box.seq.load(std::memory_order_relaxed) != seen ||
                           (box.produced.load(std::memory_order_acquire) -
                            box.consumed.load(std::memory_order_relaxed)) <
                               LaneMailbox::kRing;
                });
            }
            if (workers_stop_.load(std::memory_order_relaxed) ||
                box.seq.load(std::memory_order_relaxed) != seen) {
                break;
            }

            hashapi::HashApiResult result;
            std::string err;
            try {
                auto& lane = *lanes_impl_[static_cast<size_t>(lane_index)];
                std::lock_guard<std::mutex> lane_lock(lane.mu);
                result = lane.hash->runBatch(req);
            } catch (const std::exception& ex) {
                err = ex.what();
            } catch (...) {
                err = "lane worker failed";
            }

            const int slot = box.produced.load(std::memory_order_relaxed) % LaneMailbox::kRing;
            box.ring[slot] = std::move(result);
            box.ring_err[slot] = std::move(err);
            box.produced.fetch_add(1, std::memory_order_release);
            box.done_seq.store(seen, std::memory_order_release);
            worker_done_cv_.notify_all();
        }
    }
}

void CudaEngine::build_groups() {
    n_groups_ = 0;
    inflight_group_ = -1;
    pipeline_batch_ = 0;
    int i = 0;
    while (i < lanes_ && n_groups_ < kMaxGroups) {
        const int take = std::min(2, lanes_ - i);
        group_lo_[n_groups_] = i;
        group_hi_[n_groups_] = i + take;
        group_ready_[n_groups_] = false;
        group_creating_[n_groups_] = false;
        group_launched_[n_groups_] = false;
        group_launch_batch_[n_groups_] = 0;
        ++n_groups_;
        i += take;
    }
}

bool CudaEngine::group_lanes_done(int group) const {
    if (group < 0 || group >= n_groups_) return true;
    const int lo = group_lo_[group];
    const int hi = group_hi_[group];
    for (int i = lo; i < hi && i < kMaxLaneMailboxes; ++i) {
        const auto& box = mailboxes_[static_cast<size_t>(i)];
        if (box.done_seq.load(std::memory_order_acquire) !=
            box.seq.load(std::memory_order_relaxed)) {
            return false;
        }
    }
    return true;
}

void CudaEngine::reap_creates() {
    for (int g = 0; g < n_groups_; ++g) {
        if (!group_creating_[g] || !group_lanes_done(g)) continue;
        const int lo = group_lo_[g];
        const int hi = group_hi_[g];
        for (int i = lo; i < hi; ++i) {
            if (!lane_errors_[static_cast<size_t>(i)].empty()) {
                throw std::runtime_error(lane_errors_[static_cast<size_t>(i)]);
            }
        }
        group_creating_[g] = false;
        group_ready_[g] = true;
    }
}

int CudaEngine::pick_ready(int exclude) const {
    for (int g = 0; g < n_groups_; ++g) {
        if (g == exclude || group_launched_[g] || !group_ready_[g]) continue;
        return g;
    }
    return -1;
}

void CudaEngine::dispatch_group(int group, int role) {
    if (group < 0 || group >= n_groups_) return;
    const int lo = group_lo_[group];
    const int hi = group_hi_[group];
    {
        std::lock_guard<std::mutex> lock(worker_mu_);
        for (int i = lo; i < hi && i < kMaxLaneMailboxes; ++i) {
            auto& box = mailboxes_[static_cast<size_t>(i)];
            box.role = role;
            box.salt = proto_salt_;
            box.difficulty = proto_difficulty_;
            box.batch_size = proto_batch_size_;
            box.arena_batch_size = proto_arena_batch_;
            box.xuni_lanes = proto_xuni_lanes_;
            box.allow_xuni_base = proto_allow_xuni_base_;
            box.device_id = proto_device_id_;
            box.work_patches = proto_work_patches_;
            box.seq.fetch_add(1, std::memory_order_release);
        }
    }
    worker_cv_.notify_all();
}

void CudaEngine::wait_group(int group) {
    if (group < 0 || group >= n_groups_) return;
    const int lo = group_lo_[group];
    const int hi = group_hi_[group];
    std::unique_lock<std::mutex> lock(worker_mu_);
    const bool done = worker_done_cv_.wait_for(lock, std::chrono::seconds(8), [&] {
        for (int i = lo; i < hi && i < kMaxLaneMailboxes; ++i) {
            auto& box = mailboxes_[static_cast<size_t>(i)];
            if (box.done_seq.load(std::memory_order_acquire) !=
                box.seq.load(std::memory_order_relaxed)) {
                return false;
            }
        }
        return true;
    });
    if (!done) {
        throw std::runtime_error("lane group wait timed out");
    }
    for (int i = lo; i < hi; ++i) {
        if (!lane_errors_[static_cast<size_t>(i)].empty()) {
            throw std::runtime_error(lane_errors_[static_cast<size_t>(i)]);
        }
    }
}

void CudaEngine::ensure_lane_workers() {
    if (lanes_ <= 1) {
        if (!lane_workers_.empty()) stop_lane_workers();
        return;
    }
    if (static_cast<int>(lane_workers_.size()) == lanes_ && worker_target_lanes_ == lanes_) {
        return;
    }
    stop_lane_workers();
    worker_target_lanes_ = lanes_;
    workers_stop_ = false;
    lane_results_.assign(static_cast<size_t>(lanes_), {});
    lane_errors_.assign(static_cast<size_t>(lanes_), {});
    for (auto& box : mailboxes_) {
        box.seq.store(0, std::memory_order_relaxed);
        box.done_seq.store(0, std::memory_order_relaxed);
        box.produced.store(0, std::memory_order_relaxed);
        box.consumed.store(0, std::memory_order_relaxed);
        box.batch_size = 0;
        box.difficulty = 0;
        box.salt.clear();
    }
    lane_workers_.reserve(static_cast<size_t>(lanes_));
    for (int i = 0; i < lanes_; ++i) {
        lane_workers_.emplace_back([this, i] { lane_worker_loop(i); });
    }
}

void CudaEngine::replan(int difficulty) {
    difficulty_ = difficulty;
    if (!vram_caps_) {
        int total_mib = static_cast<int>(total_vram_bytes_ / (1024 * 1024));
        vram_caps_ = resolve_vram_caps(
            total_mib, settings_.target_vram_pct, settings_.desktop_headroom_pct,
            settings_.emergency_vram_pct, settings_.min_headroom_pct, settings_.runtime_overhead_pct,
            settings_.min_headroom_floor_mib, settings_.runtime_overhead_floor_mib,
            settings_.target_vram_mib, settings_.headroom_mib, settings_.emergency_vram_mib,
            settings_.min_headroom_mib, settings_.cuda_runtime_overhead_mib);
    }

    auto plan = plan_cuda_batch(
        total_vram_bytes_, free_vram_bytes(), vram_caps_->target_mib, vram_caps_->headroom_mib,
        difficulty, settings_.vram_reference_difficulty, max_lanes_cap_, settings_.cuda_lane_reserve,
        settings_.cuda_batch_size, settings_.cuda_max_batch_size, vram_caps_->runtime_overhead_mib);

    if (plan.batch_per_lane <= 0) {
        throw std::runtime_error("Could not size CUDA batch for difficulty " +
                                 std::to_string(difficulty));
    }
    vram_plan_ = plan;
    lanes_ = plan.lanes;
    // One full hash pack. work_patches only means "how many key bags to stage",
    // not a second Argon2 arena — that would cut the live batch in half.
    work_patches_ = settings_.work_patches;
    if (work_patches_ < 2) work_patches_ = 2;
    if (work_patches_ > 3) work_patches_ = 3;
    planned_batch_per_lane_ = plan.batch_per_lane;
    batch_per_lane_ = apply_batch_scale(plan.batch_per_lane, thermal_batch_scale_);
    hash_patch_ = 0;
    thermal_hunt_ = {};
    thermal_idle_ms_ = 0;
    build_groups();
    if (started_) sync_lanes();
}

void CudaEngine::start() {
    if (started_) return;

    // Flags must be set before the device context is created.
    // Spin: minimize host latency between multi-lane kernels. MapHost: pinned DMA.
    cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost);
    cudaError_t st = cudaSetDevice(settings_.device_id);
    if (st != cudaSuccess) {
        throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(st));
    }
    // Prefer L1 for the large Argon2 global working set.
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, settings_.device_id) == cudaSuccess) {
        device_name_ = prop.name;
        total_vram_bytes_ = prop.totalGlobalMem;
        // Ampere+: keep persisting-L2 carveout at 0 so random Argon2 ref loads
        // get the full normal L2 (streaming loads do the rest).
        if (prop.persistingL2CacheMaxSize > 0) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
        }
        if (prop.major >= 8) {
            cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 128);
        }
    } else {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        total_vram_bytes_ = total_b;
        device_name_ = "CUDA GPU";
    }

    int total_mib = static_cast<int>(total_vram_bytes_ / (1024 * 1024));
    vram_caps_ = resolve_vram_caps(
        total_mib, settings_.target_vram_pct, settings_.desktop_headroom_pct,
        settings_.emergency_vram_pct, settings_.min_headroom_pct, settings_.runtime_overhead_pct,
        settings_.min_headroom_floor_mib, settings_.runtime_overhead_floor_mib,
        settings_.target_vram_mib, settings_.headroom_mib, settings_.emergency_vram_mib,
        settings_.min_headroom_mib, settings_.cuda_runtime_overhead_mib);

    hashapi::configureKeygenPool(settings_.keygen_threads);
    replan(difficulty_);
    sync_lanes();
    ensure_lane_workers();
    started_ = true;
}

void CudaEngine::stop() {
    stop_lane_workers();
    lanes_impl_.clear();
    started_ = false;
    vram_plan_.reset();
}

void CudaEngine::set_difficulty(int difficulty) {
    if (difficulty == difficulty_ && started_) return;
    difficulty_ = difficulty;
    if (started_) {
        replan(difficulty);
        ensure_lane_workers();
    }
}

int CudaEngine::reduce_lane_cap() {
    // Do not replan — refilling VRAM on fewer lanes (8x7k → 7x9k) keeps heat high.
    max_lanes_cap_ = std::max(1, max_lanes_cap_ - 1);
    lanes_ = std::min(lanes_, max_lanes_cap_);
    if (started_) ensure_lane_workers();
    return max_lanes_cap_;
}

int CudaEngine::add_lane_cap() {
    if (max_lanes_cap_ >= config_max_lanes_) return max_lanes_cap_;
    max_lanes_cap_++;
    lanes_ = std::min(std::max(lanes_, max_lanes_cap_), config_max_lanes_);
    if (vram_plan_ && vram_plan_->lanes > 0) lanes_ = std::min(lanes_, vram_plan_->lanes);
    lanes_ = std::min(lanes_, max_lanes_cap_);
    if (started_) ensure_lane_workers();
    return max_lanes_cap_;
}

bool CudaEngine::restore_lane_cap_if_cool(int temperature_c, int /*difficulty*/,
                                          int memory_junction_c) {
    if (max_lanes_cap_ >= config_max_lanes_) return false;
    // Hybrid m=100 is when the full lane pack is needed. Restore after a thermal dip.
    if (temperature_c >= settings_.warn_gpu_temp_c) return false;
    if (settings_.thermal_use_memory_junction && memory_junction_c > 0 &&
        memory_junction_c >= settings_.warn_mem_temp_c)
        return false;
    max_lanes_cap_ = config_max_lanes_;
    if (started_) {
        replan(difficulty_);
        ensure_lane_workers();
    }
    return true;
}

double CudaEngine::set_thermal_batch_scale(double scale) {
    double floor = clamp_float(settings_.gpu_thermal_batch_min_scale, 0.20, 1.0);
    double new_scale = clamp_float(scale, floor, 1.0);
    if (std::abs(new_scale - thermal_batch_scale_) < 0.005) return thermal_batch_scale_;
    thermal_batch_scale_ = new_scale;
    if (planned_batch_per_lane_ > 0) {
        batch_per_lane_ = apply_batch_scale(planned_batch_per_lane_, thermal_batch_scale_);
        thermal_hunt_.batch = batch_per_lane_;
    }
    return thermal_batch_scale_;
}

ThermalHuntResult CudaEngine::update_thermal_batch_from_temp(int temperature_c,
                                                             int memory_junction_c, double now_s) {
    ThermalHuntResult r;
    r.batch = batch_per_lane_;
    r.scale = thermal_batch_scale_;
    r.control_c = memory_junction_c > 0 ? memory_junction_c : temperature_c;
    r.sensor = memory_junction_c > 0 ? "mem" : "gpu";
    if (!settings_.gpu_thermal_batch_enabled || planned_batch_per_lane_ <= 0) return r;

    if (thermal_hunt_.batch <= 0) {
        const double start_sc = clamp_float(settings_.gpu_thermal_start_scale, 0.50, 1.0);
        thermal_hunt_.batch = apply_batch_scale(planned_batch_per_lane_, start_sc);
        thermal_hunt_.floor = thermal_hunt_.batch;
        thermal_hunt_.last_ok = thermal_hunt_.batch;
        batch_per_lane_ = thermal_hunt_.batch;
        thermal_batch_scale_ = start_sc;
    }
    const int min_batch =
        apply_batch_scale(planned_batch_per_lane_, settings_.gpu_thermal_batch_min_scale);
    const double settle = static_cast<double>(settings_.thermal_settle_s);
    // Climb ceiling is the full VRAM pack. Floor starts at start_scale and only rises.
    r = hunt_thermal_scale(thermal_hunt_, temperature_c, memory_junction_c,
                           settings_.thermal_use_memory_junction, settings_.warn_mem_temp_c,
                           settings_.max_mem_temp_c, settings_.max_gpu_temp_c,
                           planned_batch_per_lane_, settings_.thermal_batch_step, min_batch, now_s,
                           settle, settle + 30.0);
    if (r.batch > 0 && r.batch != batch_per_lane_) {
        batch_per_lane_ = r.batch;
        thermal_batch_scale_ =
            static_cast<double>(batch_per_lane_) / static_cast<double>(planned_batch_per_lane_);
    }
    thermal_idle_ms_ = r.idle_ms;
    r.scale = thermal_batch_scale_;
    r.batch = batch_per_lane_;
    return r;
}

std::optional<BlockHit> CudaEngine::hit_from_match(const std::string& key, const std::string& hash,
                                                   const std::string& pattern, int64_t attempts,
                                                   double hps) const {
    std::string kind = classify_block(hash, pattern);
    if (kind == "OTHER") return std::nullopt;
    if (!is_hex64_key(key)) return std::nullopt;
    BlockHit hit;
    hit.key = key;
    hit.hash_str = hash;
    hit.block_type = kind;
    hit.attempts = attempts;
    hit.strategy = settings_.strategy;
    hit.hps = hps;
    hit.found_at = now_iso_local();
    hit.memory_cost = difficulty_;
    return hit;
}

namespace {

int hit_rank(const BlockHit& h) {
    if (h.block_type == "XNM") return 0;
    if (h.block_type == "XBLK") return 1;
    if (h.block_type == "XUNI") return 2;
    return 3;
}

}  // namespace

void CudaEngine::absorb_group(MineBatchResult& out, double& total_hs, int group) {
    if (group < 0 || group >= n_groups_) return;
    const int lo = group_lo_[group];
    const int hi = group_hi_[group];
    for (int i = lo; i < hi; ++i) {
        const auto& result = lane_results_[static_cast<size_t>(i)];
        if (!result.ok) {
            throw std::runtime_error(result.error.empty() ? "CUDA batch failed" : result.error);
        }
        out.hashes_done += static_cast<int64_t>(result.attempts);
        total_hs += result.hashrate;
        for (const auto& m : result.matches) {
            auto hit = hit_from_match(m.key, m.hash, m.matched_pattern,
                                      static_cast<int64_t>(result.attempts), result.hashrate);
            if (!hit) continue;
            hit->hps = total_hs > 0 ? total_hs : result.hashrate;
            out.hits.push_back(*hit);
            if (!out.hit || hit_rank(*hit) < hit_rank(*out.hit)) out.hit = hit;
        }
    }
}

MineBatchResult CudaEngine::mine_batch() {
    if (!started_ || batch_per_lane_ <= 0 || lanes_impl_.empty()) return {};

    // Always target XEN11 → XNM (common) and XBLK (rare superblocks). XUNI is optional.
    const bool allow_xuni_base = compute_allow_xuni_base();
    last_allow_xuni_ = allow_xuni_base;
    // Taper minute: at most one XUNI lane even if config allows more.
    int xuni_lanes = std::max(0, settings_.xuni_max_lanes);
    if (allow_xuni_base && in_xuni_taper_window()) xuni_lanes = std::min(xuni_lanes, 1);
    const std::string salt = settings_.salt_hex();

    MineBatchResult out;
    double total_hs = 0.0;

    auto absorb = [&](const hashapi::HashApiResult& result) {
        if (!result.ok) {
            throw std::runtime_error(result.error.empty() ? "CUDA batch failed" : result.error);
        }
        out.hashes_done += static_cast<int64_t>(result.attempts);
        total_hs += result.hashrate;
        // Keep every match (multi-hit); rank only for out.hit convenience field.
        for (const auto& m : result.matches) {
            auto hit = hit_from_match(m.key, m.hash, m.matched_pattern,
                                      static_cast<int64_t>(result.attempts), result.hashrate);
            if (!hit) continue;
            hit->hps = total_hs > 0 ? total_hs : result.hashrate;
            out.hits.push_back(*hit);
            if (!out.hit || hit_rank(*hit) < hit_rank(*out.hit)) out.hit = hit;
        }
    };

    if (lanes_ == 1) {
        hashapi::HashApiRequest req;
        req.backend = "cuda";
        req.salt_hex = salt;
        req.key_prefix = lane_prefix(0);
        req.target_pattern = "XEN11";
        req.difficulty = static_cast<uint32_t>(difficulty_);
        req.batch_size = static_cast<size_t>(batch_per_lane_);
        req.arena_batch_size = static_cast<size_t>(std::max(planned_batch_per_lane_, batch_per_lane_));
        req.allow_xuni = allow_xuni_base && (0 < xuni_lanes);
        req.gpu_first_blocks = true;
        req.first_block_dynamic_chunk_auto = false;
        req.device_id = settings_.device_id;
        req.work_patches = work_patches_;
        req.hash_patch = hash_patch_;
        req.prepare_patch = (work_patches_ > 1) ? (hash_patch_ + 1) % work_patches_ : 0;
        auto& lane = *lanes_impl_[0];
        std::lock_guard<std::mutex> lock(lane.mu);
        absorb(lane.hash->runBatch(req));
    } else {
        // Streaming 8-wide: workers keep the next wave on the GPU while we
        // absorb the last result. Kick only when salt/diff/batch/xuni change.
        ensure_lane_workers();
        bool kick = false;
        {
            std::lock_guard<std::mutex> lock(worker_mu_);
            for (int i = 0; i < lanes_ && i < kMaxLaneMailboxes; ++i) {
                auto& box = mailboxes_[static_cast<size_t>(i)];
                const bool changed =
                    box.salt != salt || box.difficulty != static_cast<uint32_t>(difficulty_) ||
                    box.batch_size != static_cast<size_t>(batch_per_lane_) ||
                    box.xuni_lanes != xuni_lanes || box.allow_xuni_base != allow_xuni_base ||
                    box.seq.load(std::memory_order_relaxed) == 0;
                if (changed) {
                    box.role = 0;
                    box.salt = salt;
                    box.difficulty = static_cast<uint32_t>(difficulty_);
                    box.batch_size = static_cast<size_t>(batch_per_lane_);
                    box.arena_batch_size = static_cast<size_t>(batch_per_lane_);
                    box.xuni_lanes = xuni_lanes;
                    box.allow_xuni_base = allow_xuni_base;
                    box.device_id = settings_.device_id;
                    box.work_patches = 0;
                    box.seq.fetch_add(1, std::memory_order_release);
                    kick = true;
                }
            }
        }
        if (kick) worker_cv_.notify_all();
        {
            std::unique_lock<std::mutex> lock(worker_mu_);
            worker_done_cv_.wait(lock, [&] {
                for (int i = 0; i < lanes_ && i < kMaxLaneMailboxes; ++i) {
                    auto& box = mailboxes_[static_cast<size_t>(i)];
                    if (box.produced.load(std::memory_order_acquire) -
                            box.consumed.load(std::memory_order_relaxed) <
                        1) {
                        return false;
                    }
                }
                return true;
            });
        }
        for (int i = 0; i < lanes_; ++i) {
            auto& box = mailboxes_[static_cast<size_t>(i)];
            const int slot = box.consumed.load(std::memory_order_relaxed) % LaneMailbox::kRing;
            if (!box.ring_err[slot].empty()) {
                throw std::runtime_error(box.ring_err[slot]);
            }
            absorb(box.ring[slot]);
            box.consumed.fetch_add(1, std::memory_order_release);
        }
        worker_done_cv_.notify_all();
    }
    last_hashrate_ = total_hs;
    for (auto& h : out.hits) h.hps = total_hs;
    if (out.hit) out.hit->hps = total_hs;
    // Stable order: better blocks first, then older attempts.
    std::sort(out.hits.begin(), out.hits.end(), [](const BlockHit& a, const BlockHit& b) {
        int ra = hit_rank(a), rb = hit_rank(b);
        if (ra != rb) return ra < rb;
        return a.key < b.key;
    });
    return out;
}

MineBatchResult CudaEngine::drain_pipeline() {
    MineBatchResult out;
    if (!started_ || lanes_impl_.empty()) return out;
    double total_hs = last_hashrate_;
    for (auto& lane : lanes_impl_) {
        std::lock_guard<std::mutex> lock(lane->mu);
        auto result = lane->hash->drainPipeline();
        if (!result.ok && !result.error.empty()) continue;
        out.hashes_done += static_cast<int64_t>(result.attempts);
        for (const auto& m : result.matches) {
            auto hit = hit_from_match(m.key, m.hash, m.matched_pattern,
                                      static_cast<int64_t>(result.attempts), total_hs);
            if (!hit) continue;
            hit->hps = total_hs;
            out.hits.push_back(*hit);
            if (!out.hit || hit_rank(*hit) < hit_rank(*out.hit)) out.hit = hit;
        }
    }
    if (out.hit) out.hit->hps = total_hs;
    return out;
}

}  // namespace xn
