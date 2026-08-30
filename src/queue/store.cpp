#include "queue/store.hpp"

#include "queue/policy.hpp"
#include "util/paths.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>

namespace xn {
namespace {

using json = nlohmann::json;

json hit_to_json(const BlockHit& hit) {
    json j = {{"key", hit.key},
              {"hash_str", hit.hash_str},
              {"block_type", hit.block_type},
              {"strategy", hit.strategy},
              {"attempts", hit.attempts},
              {"hps", hit.hps},
              {"found_at", hit.found_at}};
    if (hit.memory_cost) j["memory_cost"] = *hit.memory_cost;
    return j;
}

BlockHit hit_from_json(const json& j) {
    BlockHit hit;
    hit.key = j.value("key", "");
    hit.hash_str = j.value("hash_str", "");
    hit.block_type = j.value("block_type", "XNM");
    hit.strategy = j.value("strategy", "random");
    hit.attempts = j.value("attempts", 0);
    hit.hps = j.value("hps", 0.0);
    hit.found_at = j.value("found_at", "");
    if (j.contains("memory_cost") && !j["memory_cost"].is_null()) {
        hit.memory_cost = j["memory_cost"].get<int>();
    }
    return hit;
}

}  // namespace

std::string hits_to_bag_json(const std::vector<BlockHit>& hits, const std::string& worker) {
    json root;
    root["worker"] = worker;
    root["hits"] = json::array();
    for (const auto& hit : hits) root["hits"].push_back(hit_to_json(hit));
    return root.dump();
}

int hits_from_bag_json(const std::string& body, std::vector<BlockHit>& out, std::string* worker) {
    out.clear();
    json root = json::parse(body, nullptr, false);
    if (root.is_discarded()) return 0;
    if (worker) {
        if (root.contains("worker")) *worker = root.value("worker", "");
        else if (root.contains("source")) *worker = root.value("source", "");
    }
    auto take = [&](const json& item) {
        if (!item.is_object()) return;
        BlockHit hit = hit_from_json(item);
        if (hit.hash_str.empty() && hit.key.empty()) return;
        out.push_back(std::move(hit));
    };
    if (root.contains("hits") && root["hits"].is_array()) {
        for (const auto& item : root["hits"]) take(item);
    } else if (root.contains("pending") && root["pending"].is_array()) {
        for (const auto& item : root["pending"]) take(item);
    } else if (root.is_object() && (root.contains("hash_str") || root.contains("key"))) {
        take(root);
    }
    return static_cast<int>(out.size());
}

BlockStore::BlockStore(std::filesystem::path db_path, std::filesystem::path jsonl_path,
                       std::filesystem::path rejected_jsonl_path)
    : db_path_(std::move(db_path)),
      jsonl_path_(std::move(jsonl_path)),
      rejected_jsonl_path_(std::move(rejected_jsonl_path)) {
    ensure_parent_dir(db_path_);
    ensure_parent_dir(jsonl_path_);
    ensure_parent_dir(rejected_jsonl_path_);
    load();
}

BlockStore::~BlockStore() {
    std::lock_guard<std::mutex> lock(mu_);
    if (dirty_ops_ > 0) save_db_unlocked();
}

void BlockStore::load() {
    pending_.clear();
    hash_index_.clear();
    next_id_ = 1;
    dirty_ops_ = 0;
    std::ifstream in(db_path_);
    if (in) try {
        json root = json::parse(in, nullptr, false);
        if (root.is_discarded()) {
            recover_jsonl_unlocked();
            return;
        }
        next_id_ = root.value("next_id", 1);
        if (root.contains("pending") && root["pending"].is_array()) {
            for (const auto& item : root["pending"]) {
                PendingBlock pb;
                pb.id = item.value("id", next_id_++);
                pb.hit = hit_from_json(item);
                pb.queue_reason = item.value("queue_reason", item.value("reason", ""));
                pb.reject_count = item.value("reject_count", 0);
                pb.forwarded = item.value("forwarded", false);
                next_id_ = std::max(next_id_, pb.id + 1);
                if (!pb.hit.hash_str.empty()) hash_index_.insert(pb.hit.hash_str);
                pending_.push_back(std::move(pb));
            }
        }
    } catch (...) {
        // Keep whatever we parsed; jsonl recovery can refill.
    }
    recover_jsonl_unlocked();
}

void BlockStore::recover_jsonl_unlocked() {
    std::ifstream in(jsonl_path_);
    if (!in) return;
    bool added = false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        json rec = json::parse(line, nullptr, false);
        if (rec.is_discarded() || !rec.is_object()) continue;
        BlockHit hit = hit_from_json(rec);
        if (hit.hash_str.empty() || hash_index_.count(hit.hash_str)) continue;
        PendingBlock pb;
        pb.id = rec.value("id", next_id_);
        if (pb.id < 1) pb.id = next_id_;
        for (const auto& existing : pending_) {
            if (existing.id == pb.id) {
                pb.id = next_id_;
                break;
            }
        }
        pb.hit = std::move(hit);
        pb.queue_reason = rec.value("queue_reason", rec.value("reason", "jsonl_recover"));
        pb.reject_count = rec.value("reject_count", 0);
        next_id_ = std::max(next_id_, pb.id + 1);
        hash_index_.insert(pb.hit.hash_str);
        pending_.push_back(std::move(pb));
        added = true;
    }
    if (added) save_db_unlocked();
}

void BlockStore::save_db_unlocked() {
    json root;
    root["next_id"] = next_id_;
    root["pending"] = json::array();
    for (const auto& pb : pending_) {
        json item = hit_to_json(pb.hit);
        item["id"] = pb.id;
        item["queue_reason"] = pb.queue_reason;
        item["reject_count"] = pb.reject_count;
        item["forwarded"] = pb.forwarded;
        root["pending"].push_back(std::move(item));
    }
    std::ofstream out(db_path_, std::ios::trunc);
    // Compact JSON — hot path can rewrite large bags; pretty-print is pure overhead.
    out << root.dump();
    dirty_ops_ = 0;
}

void BlockStore::mark_dirty_unlocked() { ++dirty_ops_; }

void BlockStore::maybe_save_unlocked() {
    if (dirty_ops_ >= kSaveEveryOps) save_db_unlocked();
}

void BlockStore::flush() {
    std::lock_guard<std::mutex> lock(mu_);
    if (dirty_ops_ > 0) save_db_unlocked();
}

void BlockStore::append_queue_jsonl(const PendingBlock& pb) {
    json rec = hit_to_json(pb.hit);
    rec["queued_at"] = now_iso_local();
    rec["reason"] = pb.queue_reason;
    rec["id"] = pb.id;
    std::ofstream out(jsonl_path_, std::ios::app);
    out << rec.dump() << "\n";
}

int64_t BlockStore::enqueue_unlocked(const BlockHit& hit, const std::string& reason,
                                     bool write_jsonl) {
    if (!hit.hash_str.empty() && hash_index_.count(hit.hash_str)) {
        return 0;  // same digest already bagged (wrong-key twin)
    }
    PendingBlock pb;
    pb.id = next_id_++;
    pb.hit = hit;
    pb.queue_reason = reason;
    pb.forwarded = false;
    if (!hit.hash_str.empty()) hash_index_.insert(hit.hash_str);
    if (write_jsonl) append_queue_jsonl(pb);
    pending_.push_back(std::move(pb));
    mark_dirty_unlocked();
    maybe_save_unlocked();
    return pending_.back().id;
}

int64_t BlockStore::enqueue(const BlockHit& hit, const std::string& reason) {
    std::lock_guard<std::mutex> lock(mu_);
    return enqueue_unlocked(hit, reason, true);
}

std::vector<PendingBlock> BlockStore::list_unforwarded(int limit) {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<PendingBlock> out;
    for (const auto& pb : pending_) {
        if (pb.forwarded) continue;
        out.push_back(pb);
        if (limit > 0 && static_cast<int>(out.size()) >= limit) break;
    }
    return out;
}

void BlockStore::mark_forwarded_many(const std::vector<int64_t>& ids) {
    if (ids.empty()) return;
    std::lock_guard<std::mutex> lock(mu_);
    for (int64_t id : ids) {
        for (auto& pb : pending_) {
            if (pb.id == id) {
                pb.forwarded = true;
                break;
            }
        }
    }
    mark_dirty_unlocked();
    save_db_unlocked();
}

void BlockStore::log_rejection(const BlockHit& hit, int http_status, const std::string& body,
                               const std::string& source) {
    json rec = hit_to_json(hit);
    rec["rejected_at"] = now_iso_local();
    rec["http_status"] = http_status;
    rec["response_body"] = body;
    rec["source"] = source;
    rec["category"] = "resubmission";
    std::ofstream out(rejected_jsonl_path_, std::ios::app);
    out << rec.dump() << "\n";
}

void BlockStore::record_direct_submit(const BlockHit& hit, int http_status,
                                      const std::string& body) {
    // Optional audit trail in rejected/jsonl-style; keep lightweight.
    (void)hit;
    (void)http_status;
    (void)body;
}

bool BlockStore::record_rejection(const BlockHit& hit, int http_status, const std::string& body,
                                  const std::string& source, const std::string& reason) {
    log_rejection(hit, http_status, body, source);
    enqueue(hit, reason);
    return true;
}

void BlockStore::mark_submitted(int64_t id, int http_status, const std::string& body) {
    (void)http_status;
    (void)body;
    mark_submitted_many(std::vector<int64_t>{id});
}

void BlockStore::mark_submitted_many(const std::vector<int64_t>& ids) {
    if (ids.empty()) return;
    std::lock_guard<std::mutex> lock(mu_);
    // Small id sets: linear scan is fine; large waves still one rewrite.
    for (int64_t id : ids) {
        pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
                                      [this, id](const PendingBlock& pb) {
                                          if (pb.id != id) return false;
                                          if (!pb.hit.hash_str.empty()) hash_index_.erase(pb.hit.hash_str);
                                          return true;
                                      }),
                       pending_.end());
    }
    mark_dirty_unlocked();
    save_db_unlocked();
}

void BlockStore::mark_pending_reason(int64_t id, const std::string& reason) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& pb : pending_) {
        if (pb.id == id) {
            pb.queue_reason = reason;
            break;
        }
    }
    mark_dirty_unlocked();
    maybe_save_unlocked();
}

int BlockStore::pending_count() {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(pending_.size());
}

int BlockStore::pending_matching_m(int m, int default_m) {
    std::lock_guard<std::mutex> lock(mu_);
    int n = 0;
    for (const auto& pb : pending_) {
        const int hit_m = pb.hit.memory_cost.value_or(default_m > 0 ? default_m : m);
        if (hit_m == m) ++n;
    }
    return n;
}

int BlockStore::pending_eligible_m(int net_m, int default_m) {
    std::lock_guard<std::mutex> lock(mu_);
    int n = 0;
    for (const auto& pb : pending_) {
        const int hit_m = pb.hit.memory_cost.value_or(default_m > 0 ? default_m : net_m);
        if (hit_m >= net_m) ++n;
    }
    return n;
}

std::unordered_map<std::string, int> BlockStore::pending_by_type(bool resubmission_only) {
    std::lock_guard<std::mutex> lock(mu_);
    std::unordered_map<std::string, int> counts{{"XUNI", 0}, {"XNM", 0}, {"XBLK", 0}};
    for (const auto& pb : pending_) {
        if (resubmission_only && pb.queue_reason != "resubmission" && pb.queue_reason != "parked")
            continue;
        std::string k = pb.hit.block_type;
        if (k != "XUNI" && k != "XBLK") k = "XNM";
        counts[k]++;
    }
    return counts;
}

std::vector<PendingBlock> BlockStore::list_pending() {
    std::lock_guard<std::mutex> lock(mu_);
    return pending_;
}

std::vector<PendingBlock> BlockStore::list_flush_batch(int net_m, int default_m, int limit,
                                                       int64_t skip_before_id) {
    std::lock_guard<std::mutex> lock(mu_);
    const int cap = limit > 0 ? limit : static_cast<int>(pending_.size());
    std::vector<PendingBlock> preferred;
    std::vector<PendingBlock> wrapped;
    preferred.reserve(static_cast<size_t>(std::max(cap, 0)));
    wrapped.reserve(static_cast<size_t>(std::max(cap, 0)));
    auto take = [&](std::vector<PendingBlock>& dest, const PendingBlock& pb) {
        if (static_cast<int>(dest.size()) >= cap) return;
        const int hit_m = pb.hit.memory_cost.value_or(default_m > 0 ? default_m : net_m);
        // Eligibility: hit m >= net m. The server rejects only strictly-lower m
        // (submitted_m < current), so higher-m finds parked during a difficulty spike
        // become submittable as soon as difficulty falls to/below their m.
        if (hit_m < net_m) return;
        // Oldest bag is often XUNI (outside_xuni_window). A tiny probe wave
        // used to fill with those and report "0 eligible" while 180k XNM sat behind.
        if (!ready_to_flush(pb.hit.block_type).first) return;
        dest.push_back(pb);
    };
    for (const auto& pb : pending_) {
        if (skip_before_id > 0 && pb.id < skip_before_id) take(wrapped, pb);
        else take(preferred, pb);
        if (static_cast<int>(preferred.size()) >= cap) break;
    }
    if (static_cast<int>(preferred.size()) < cap) {
        for (auto& pb : wrapped) {
            preferred.push_back(std::move(pb));
            if (static_cast<int>(preferred.size()) >= cap) break;
        }
    }
    std::sort(preferred.begin(), preferred.end(), [](const PendingBlock& a, const PendingBlock& b) {
        int pa = flush_priority(a.hit.block_type);
        int pb = flush_priority(b.hit.block_type);
        if (pa != pb) return pa < pb;
        return a.id < b.id;
    });
    return preferred;
}

void BlockStore::defer_all_to_next_start() {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& pb : pending_) {
        if (pb.hit.block_type == "XUNI" && !in_xuni_submit_window()) {
            pb.queue_reason = OUTSIDE_XUNI_WINDOW_REASON;
        } else {
            pb.queue_reason = SHUTDOWN_PENDING_REASON;
        }
    }
    mark_dirty_unlocked();
    save_db_unlocked();
}

}  // namespace xn
