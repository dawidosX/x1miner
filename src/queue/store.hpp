#pragma once

#include "common.hpp"

#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace xn {

struct PendingBlock {
    int64_t id = 0;
    BlockHit hit;
    std::string queue_reason;
    int reject_count = 0;
    bool forwarded = false;
};

std::string hits_to_bag_json(const std::vector<BlockHit>& hits, const std::string& worker);
int hits_from_bag_json(const std::string& body, std::vector<BlockHit>& out, std::string* worker);

class BlockStore {
public:
    BlockStore(std::filesystem::path db_path, std::filesystem::path jsonl_path,
               std::filesystem::path rejected_jsonl_path);
    ~BlockStore();

    int64_t enqueue(const BlockHit& hit, const std::string& reason);
    void log_rejection(const BlockHit& hit, int http_status, const std::string& body,
                       const std::string& source);
    void record_direct_submit(const BlockHit& hit, int http_status, const std::string& body);
    // Returns true if newly queued for resubmit.
    // reason: "resubmission" (generic retry) or "parked" (401 difficulty — retry when net_m <= block m).
    bool record_rejection(const BlockHit& hit, int http_status, const std::string& body,
                          const std::string& source, const std::string& reason = "resubmission");
    void mark_submitted(int64_t id, int http_status, const std::string& body);
    /// Batch remove after a flush wave — one disk rewrite instead of N.
    void mark_submitted_many(const std::vector<int64_t>& ids);
    void mark_pending_reason(int64_t id, const std::string& reason);
    /// Force compact JSON snapshot to disk (call on shutdown / after flush waves).
    void flush();

    int pending_count();
    // Count pending blocks whose hit.memory_cost matches m (or missing cost treated as default_m).
    int pending_matching_m(int m, int default_m = 0);
    // Count pending blocks eligible for submit at net_m: hit m >= net_m.
    // Server rejects only strictly-lower m (submitted_m < current) — equal-or-higher passes,
    // so higher-m "parked" finds become submittable whenever difficulty falls to/below their m.
    int pending_eligible_m(int net_m, int default_m = 0);
    std::unordered_map<std::string, int> pending_by_type(bool resubmission_only = false);
    std::vector<PendingBlock> list_pending();
    /// Oldest matching-m hits that are ready to flush (skips XUNI outside window).
    /// XNM then XBLK then XUNI. limit<=0 = all matching.
    /// skip_before_id: prefer ids >= this (wrap to older ids if the wave is short).
    std::vector<PendingBlock> list_flush_batch(int net_m, int default_m, int limit,
                                               int64_t skip_before_id = 0);
    std::vector<PendingBlock> list_unforwarded(int limit = 0);
    void mark_forwarded_many(const std::vector<int64_t>& ids);
    void defer_all_to_next_start();

private:
    void load();
    void recover_jsonl_unlocked();
    void save_db_unlocked();
    void mark_dirty_unlocked();
    void maybe_save_unlocked();
    void append_queue_jsonl(const PendingBlock& pb);
    int64_t enqueue_unlocked(const BlockHit& hit, const std::string& reason, bool write_jsonl);

    std::filesystem::path db_path_;
    std::filesystem::path jsonl_path_;
    std::filesystem::path rejected_jsonl_path_;
    std::mutex mu_;
    std::vector<PendingBlock> pending_;
    std::unordered_set<std::string> hash_index_;
    int64_t next_id_ = 1;
    int dirty_ops_ = 0;
    static constexpr int kSaveEveryOps = 8;
};

}  // namespace xn
