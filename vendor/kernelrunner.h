#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class KernelRunner
{
private:
    uint32_t type, version;
    uint32_t passes, lanes, segmentBlocks;
    uint32_t allocatedSegmentBlocks;
    std::size_t batchSize;
    /// Warps (Argon2 jobs) packed per CTA — 4 is stable (higher → CUDA 701).
    uint32_t jobsPerBlock = 4;
    int deviceMajor = 0;
    int deviceMinor = 0;

    cudaEvent_t start, end, copyStart, copyEnd, firstBlockStart, firstBlockEnd, kernelStart, kernelEnd;
    cudaEvent_t finalizeStart, finalizeEnd;
    cudaStream_t stream;
    void* memory;
    void* refs;
    void* deviceKeys;
    void* deviceSalt;
    void* deviceFinalHashes;          // batch * 64 — GPU finalize digests
    std::size_t deviceKeysCapacity;
    std::size_t deviceSaltCapacity;
    std::size_t deviceFinalHashesCapacity;
    bool deviceFirstBlocksReady;
    std::size_t deviceFirstBlockKeyLength;
    std::uint32_t deviceFirstBlockSaltLength;
    std::uint32_t deviceFirstBlockOutputLength;
    std::uint32_t deviceFirstBlockMemoryCost;
    std::uint32_t deviceFirstBlockTimeCost;
    std::uint32_t deviceFirstBlockVersion;
    std::uint32_t deviceFirstBlockType;
    std::uint32_t deviceFirstBlockLanes;
    bool lastUsedDeviceFirstBlocks;
    bool lastUsedDeviceFinalHashes;

    std::unique_ptr<uint8_t[]> blocksIn;   // legacy pageable (unused when pinned)
    std::unique_ptr<uint8_t[]> blocksOut;
    uint8_t* blocksInPinned = nullptr;     // cudaHostAlloc — async DMA
    uint8_t* blocksOutPinned = nullptr;    // last 1KB block/job (CPU-finalize fallback)
    uint8_t* hostFinalHashesPinned = nullptr; // 64 B digests when GPU finalize is used
    uint8_t* hostKeysPinned = nullptr;
    std::size_t hostKeysPinnedCapacity = 0;
    std::size_t hostFinalHashesCapacity = 0;

    uint8_t* inputBase() const {
        return blocksInPinned ? blocksInPinned : blocksIn.get();
    }
    uint8_t* outputBase() const {
        return blocksOutPinned ? blocksOutPinned : blocksOut.get();
    }

    void copyInputBlocks();
    void copyOutputBlocks();
    void copyFinalHashes();
    void ensureFinalHashBuffers();

    void runDeviceFirstBlockKernel();
    void runDeviceFinalizeKernel();
    void runKernelOneshot();

public:

    std::size_t getBatchSize() const { return batchSize; }

    KernelRunner(uint32_t type, uint32_t version,
        uint32_t passes, uint32_t lanes,
        uint32_t segmentBlocks, std::size_t batchSize);
    ~KernelRunner();

    void init(std::size_t batchSize);
    bool canReuse(uint32_t type, uint32_t version,
        uint32_t passes, uint32_t lanes,
        uint32_t segmentBlocks, std::size_t batchSize) const;
    void reconfigure(uint32_t type, uint32_t version,
        uint32_t passes, uint32_t lanes,
        uint32_t segmentBlocks, std::size_t batchSize);

    void* getInputMemory(std::size_t jobId) const;
    const void* getOutputMemory(std::size_t jobId) const;
    /// 64-byte digests after GPU finalize (nullptr / invalid if fallback path).
    const void* getFinalHashMemory(std::size_t jobId) const;
    bool hasDeviceFinalHashes() const { return lastUsedDeviceFinalHashes; }

    /// Ensure pinned host key arena can hold `bytes`; returns write pointer.
    uint8_t* ensureHostKeyBuffer(std::size_t bytes);
    std::size_t hostKeyBufferCapacity() const { return hostKeysPinnedCapacity; }

    bool prepareInputBlocksOnDevice(const std::vector<std::string>& passwords,
                                    const std::vector<std::uint8_t>& saltBytes,
                                    std::uint32_t outputLength,
                                    std::uint32_t memoryCost,
                                    std::uint32_t timeCost,
                                    std::uint32_t version,
                                    std::uint32_t type,
                                    std::uint32_t lanes);
    /// Keys already written into ensureHostKeyBuffer() / hostKeysPinned (flat count*keyLen).
    bool prepareInputBlocksOnDeviceFlat(std::size_t keyLength,
                                        const std::vector<std::uint8_t>& saltBytes,
                                        std::uint32_t outputLength,
                                        std::uint32_t memoryCost,
                                        std::uint32_t timeCost,
                                        std::uint32_t version,
                                        std::uint32_t type,
                                        std::uint32_t lanes);

    void run();
    float finish();
    float getLastHostToDeviceMs() const;
    float getLastGpuFirstBlockMs() const;
    float getLastDeviceToHostMs() const;
    float getLastGpuFinalizeMs() const;
};
