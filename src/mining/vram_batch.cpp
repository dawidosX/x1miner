#include "mining/vram_batch.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace xn {

std::string CudaVramPlan::summary() const {
    std::ostringstream oss;
    if (lanes > 1) {
        oss << "lanes=" << lanes << "x" << batch_per_lane;
    } else {
        oss << "batch=" << batch_size;
    }
    if (lane_reserve > 0 && lanes >= lane_reserve) {
        oss << " reserve=" << lane_reserve;
    }
    oss << " budget=" << budget_mib << "MiB"
        << " batch_vram~=" << batch_vram_mib << "MiB"
        << " cuda_overhead=" << runtime_overhead_mib << "MiB"
        << " projected_used=" << projected_used_mib << "MiB"
        << " projected_free=" << projected_headroom_mib << "MiB"
        << " (target<=" << target_mib << "MiB desktop>=" << desktop_headroom_mib << "MiB)";
    return oss.str();
}

bool CudaVramPlan::within_limits() const {
    return projected_used_mib <= target_mib && projected_headroom_mib >= desktop_headroom_mib;
}

bool CudaVramPlan::fills_budget(int tolerance_mib) const {
    return std::abs(batch_vram_mib - budget_mib) <= tolerance_mib;
}

double bytes_per_attempt(int difficulty) {
    return static_cast<double>(difficulty) * 1024.0 * BYTES_PER_ATTEMPT_FACTOR;
}

int cuda_lane_count(int difficulty, int reference_difficulty, int max_lanes) {
    if (max_lanes <= 1 || difficulty <= 0) return 1;
    // m=100 harvest: always use the configured lane pack (VRAM planner may shrink it).
    if (difficulty == HARVEST_DIFFICULTY) return std::max(1, max_lanes);
    if (reference_difficulty <= 0 || difficulty >= reference_difficulty) return 1;
    const int boost = reference_difficulty / difficulty;
    return std::max(1, std::min(max_lanes, boost));
}

uint64_t estimate_batch_vram_bytes(int batch_size, int difficulty) {
    if (batch_size <= 0 || difficulty <= 0) return 0;
    return static_cast<uint64_t>(batch_size * bytes_per_attempt(difficulty));
}

int memory_limited_batch_size(uint64_t free_vram_bytes, int difficulty, uint64_t reserve_bytes) {
    if (difficulty <= 0 || free_vram_bytes <= reserve_bytes) return 0;
    double available = static_cast<double>(free_vram_bytes - reserve_bytes);
    double per = bytes_per_attempt(difficulty);
    if (per <= 0) return 0;
    return static_cast<int>(available / per);
}

int select_batch_size(uint64_t budget_bytes, int difficulty, int explicit_max_batch,
                      bool fill_vram_cap) {
    uint64_t dll_free_arg = budget_bytes + CUDA_ENGINE_RESERVE_BYTES;
    int memory_limit = memory_limited_batch_size(dll_free_arg, difficulty);
    if (memory_limit <= 0) return 0;
    if (explicit_max_batch > 0) return std::min(memory_limit, explicit_max_batch);
    if (!fill_vram_cap) {
        int tuned = 0;
        if (difficulty <= 1) tuned = 2048;
        else if (difficulty <= 8) tuned = 4096;
        else if (difficulty <= 64) tuned = 3072;
        if (tuned > 0) return std::min(memory_limit, tuned);
    }
    return memory_limit;
}

static void project(int total_mib, int lanes, int batch_per_lane, int difficulty,
                    int runtime_overhead_mib, uint64_t& batch_vram_bytes, int& batch_vram_mib,
                    int& projected_used_mib, int& projected_headroom_mib) {
    batch_vram_bytes = estimate_batch_vram_bytes(batch_per_lane, difficulty) *
                       static_cast<uint64_t>(lanes);
    batch_vram_mib = static_cast<int>(batch_vram_bytes / (1024 * 1024));
    projected_used_mib = batch_vram_mib + runtime_overhead_mib;
    projected_headroom_mib = std::max(0, total_mib - projected_used_mib);
}

static CudaVramPlan clamp_plan_to_caps(CudaVramPlan plan) {
    if (plan.within_limits()) return plan;
    int total_mib = plan.projected_used_mib + plan.projected_headroom_mib;
    int lanes = plan.lanes;
    int batch_per_lane = plan.batch_per_lane;

    for (int i = 0; i < 10000; ++i) {
        uint64_t batch_vram_bytes = 0;
        int batch_vram_mib = 0, projected_used_mib = 0, projected_headroom_mib = 0;
        project(total_mib, lanes, batch_per_lane, plan.difficulty, plan.runtime_overhead_mib,
                batch_vram_bytes, batch_vram_mib, projected_used_mib, projected_headroom_mib);
        if (projected_used_mib <= plan.target_mib &&
            projected_headroom_mib >= plan.desktop_headroom_mib) {
            plan.batch_size = batch_per_lane;
            plan.lanes = lanes;
            plan.batch_per_lane = batch_per_lane;
            plan.batch_vram_bytes = batch_vram_bytes;
            plan.batch_vram_mib = batch_vram_mib;
            plan.projected_used_mib = projected_used_mib;
            plan.projected_headroom_mib = projected_headroom_mib;
            return plan;
        }
        if (batch_per_lane > 1) {
            batch_per_lane = std::max(1, static_cast<int>(batch_per_lane * 0.98));
            continue;
        }
        if (lanes > 1) {
            --lanes;
            uint64_t per_lane_budget = std::max<uint64_t>(1, plan.budget_bytes / lanes);
            batch_per_lane = select_batch_size(per_lane_budget, plan.difficulty, 0, true);
            continue;
        }
        break;
    }
    return plan;
}

CudaVramPlan plan_cuda_batch(uint64_t total_bytes, uint64_t free_bytes, int target_mib,
                             int desktop_headroom_mib, int difficulty, int reference_difficulty,
                             int max_lanes, int lane_reserve, int explicit_batch,
                             int explicit_max_batch, int runtime_overhead_mib) {
    int total_mib = static_cast<int>(total_bytes / (1024 * 1024));
    int used_before_mib =
        static_cast<int>(std::max<int64_t>(0, static_cast<int64_t>(total_bytes - free_bytes) /
                                                  (1024 * 1024)));

    int effective_target_mib = target_mib;
    int cap_batch_mib = std::max(0, effective_target_mib - runtime_overhead_mib);
    int headroom_limited_mib =
        std::max(0, total_mib - desktop_headroom_mib - runtime_overhead_mib);
    int allowed_mib = std::min(cap_batch_mib, headroom_limited_mib);
    uint64_t budget_bytes = static_cast<uint64_t>(allowed_mib) * 1024ULL * 1024ULL;
    int budget_mib = allowed_mib;

    int lanes = cuda_lane_count(difficulty, reference_difficulty, std::max(1, max_lanes));
    int reserve = std::max(0, lane_reserve);
    uint64_t per_lane_budget = std::max<uint64_t>(1, budget_bytes / std::max(1, lanes));

    int max_batch_per_lane =
        select_batch_size(per_lane_budget, difficulty, explicit_max_batch, true);
    int batch_per_lane = 0;
    if (max_batch_per_lane > 0) {
        batch_per_lane =
            explicit_batch > 0 ? std::min(explicit_batch, max_batch_per_lane) : max_batch_per_lane;
    }

    uint64_t batch_vram_bytes =
        estimate_batch_vram_bytes(batch_per_lane, difficulty) * static_cast<uint64_t>(lanes);
    int batch_vram_mib = static_cast<int>(batch_vram_bytes / (1024 * 1024));
    int projected_used_mib = batch_vram_mib + runtime_overhead_mib;
    int projected_headroom_mib = std::max(0, total_mib - projected_used_mib);

    CudaVramPlan plan;
    plan.batch_size = batch_per_lane;
    plan.lanes = lanes;
    plan.batch_per_lane = batch_per_lane;
    plan.lane_reserve = reserve;
    plan.budget_bytes = budget_bytes;
    plan.budget_mib = budget_mib;
    plan.batch_vram_bytes = batch_vram_bytes;
    plan.batch_vram_mib = batch_vram_mib;
    plan.used_before_mib = used_before_mib;
    plan.projected_used_mib = projected_used_mib;
    plan.projected_headroom_mib = projected_headroom_mib;
    plan.runtime_overhead_mib = runtime_overhead_mib;
    plan.target_mib = target_mib;
    plan.effective_target_mib = effective_target_mib;
    plan.desktop_headroom_mib = desktop_headroom_mib;
    plan.difficulty = difficulty;
    return clamp_plan_to_caps(plan);
}

}  // namespace xn
