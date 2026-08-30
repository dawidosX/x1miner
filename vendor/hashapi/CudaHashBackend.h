#pragma once

#include "HashApiTypes.h"

#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "../argon2params.h"

class ComputeBackend;

namespace hashapi {

/// Host-side double-buffer slot: GPU results copied out so the next batch can
/// run while base64+match of the previous batch runs on the CPU.
struct PipelinedBatch {
    bool ready = false;
    /// True when digests_ holds 64-byte Blake2b outputs (GPU finalize).
    /// False when digests_ holds 1024-byte final Argon2 blocks (CPU finalize).
    bool gpu_finalized = false;
    /// Flat keys: attempts * key_length (hex chars, no nulls).
    std::vector<std::uint8_t> keys;
    std::size_t key_length = kHashApiKeyLength;
    /// Either attempts*64 digests or attempts*1024 blocks.
    std::vector<std::uint8_t> digests;
    Argon2Params params;
    std::string target_pattern = "XEN11";
    bool allow_xuni = false;
    bool single_key = false;
    std::string request_id;
    std::size_t attempts = 0;
    std::uint32_t difficulty = 0;
};

class CudaHashBackend : public IHashBackend {
public:
    explicit CudaHashBackend(ComputeBackend& backend);
    explicit CudaHashBackend(std::unique_ptr<ComputeBackend> backend);
    ~CudaHashBackend() override;

    HashApiResult runBatch(const HashApiRequest& request) override;
    /// Finalize any buffered GPU results without launching a new batch.
    HashApiResult drainPipeline();

private:
    ComputeBackend& backend();
    const ComputeBackend& backend() const;
    void ensureInitialized(ComputeBackend& backend,
                           const Argon2Params& params,
                           std::size_t batch_size);
    void finalizeSlot(const PipelinedBatch& slot, HashApiResult& result) const;
    PipelinedBatch snapshotOutputs(const HashApiRequest& request,
                                   Argon2Params params,
                                   const std::uint8_t* keys,
                                   std::size_t key_length,
                                   bool single_key,
                                   std::size_t attempts) const;
    // Champ rule: keys that go to /verify are the keys this oneshot hashed.
    void freezeHashedKeys(const std::uint8_t* src, std::size_t attempts, std::size_t key_len);

    ComputeBackend* backend_ = nullptr;
    std::unique_ptr<ComputeBackend> owned_backend_;
    bool initialized_ = false;
    std::size_t initialized_batch_size_ = 0;
    std::uint32_t initialized_segment_blocks_ = 0;
    PipelinedBatch pending_;

    // Hot-path caches (salt/device stable across mining batches).
    std::string cached_salt_hex_;
    std::vector<std::uint8_t> cached_salt_bytes_;
    int cached_device_id_ = -1;
    bool device_info_cached_ = false;

    bool preload_ready_ = false;
    int preload_slot_ = 0;
    int live_slot_ = 0;
    std::size_t preload_attempts_ = 0;
    std::vector<std::uint8_t> next_keys_;
    std::future<void> next_keygen_;

    std::vector<std::uint8_t> hashed_keys_;
    std::size_t hashed_attempts_ = 0;
    std::size_t hashed_key_len_ = kHashApiKeyLength;
    // Keys frozen at hash-launch. Must not be overwritten until collect snapshots them.
    std::vector<std::uint8_t> bound_keys_;
    std::size_t bound_attempts_ = 0;
    bool bound_ready_ = false;
};

} // namespace hashapi
