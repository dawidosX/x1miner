#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct DeviceInfo {
	int index;
	int busId;
	std::string name;
	size_t totalMemoryBytes;
};

class ComputeBackend {
public:
	virtual ~ComputeBackend() = default;

	virtual DeviceInfo getDeviceInfo() const = 0;
	virtual size_t getFreeMemory() const = 0;

	// Activate device for current thread (e.g. cudaSetDevice)
	virtual void activate() = 0;

	// Allocate buffers for batch Argon2 hashing.
	// Can be called multiple times; previous allocations are released.
	virtual void init(size_t batchSize, uint32_t type, uint32_t version,
	                  uint32_t passes, uint32_t lanes,
	                  uint32_t segmentBlocks) = 0;

	virtual void* getInputMemory(size_t jobId) const = 0;
	virtual const void* getOutputMemory(size_t jobId) const = 0;

	/// True after a run that produced 64-byte digests on device (p=1 path).
	virtual bool hasDeviceFinalHashes() const { return false; }
	/// 64-byte digest for jobId when hasDeviceFinalHashes(); otherwise nullptr.
	virtual const void* getFinalHashMemory(size_t jobId) const
	{
		(void)jobId;
		return nullptr;
	}

	/// Grow/reuse pinned host key arena; returns write pointer (count * keyLen bytes).
	virtual uint8_t* ensureHostKeyBuffer(size_t bytes)
	{
		(void)bytes;
		return nullptr;
	}
	virtual uint8_t* ensureHostKeyBufferSlot(int slot, size_t bytes)
	{
		if (slot == 0) return ensureHostKeyBuffer(bytes);
		return nullptr;
	}
	virtual void setLiveKeySlot(int slot) { (void)slot; }
	virtual int liveKeySlot() const { return 0; }
	virtual void setWorkPatches(int n) { (void)n; }
	virtual void setHashPatch(int n) { (void)n; }
	virtual bool patchReady(int n) const {
		(void)n;
		return false;
	}
	virtual bool prepareIdlePatch(int n) {
		(void)n;
		return false;
	}
	virtual void waitPrepare() {}

	virtual bool prepareInputBlocksOnDevice(const std::vector<std::string>& passwords,
	                                        const std::vector<std::uint8_t>& saltBytes,
	                                        std::uint32_t outputLength,
	                                        std::uint32_t memoryCost,
	                                        std::uint32_t timeCost,
	                                        std::uint32_t version,
	                                        std::uint32_t type,
	                                        std::uint32_t lanes)
	{
		(void)passwords;
		(void)saltBytes;
		(void)outputLength;
		(void)memoryCost;
		(void)timeCost;
		(void)version;
		(void)type;
		(void)lanes;
		return false;
	}

	/// Keys already written into ensureHostKeyBuffer() (flat batch*keyLen).
	virtual bool prepareInputBlocksOnDeviceFlat(std::size_t keyLength,
	                                            const std::vector<std::uint8_t>& saltBytes,
	                                            std::uint32_t outputLength,
	                                            std::uint32_t memoryCost,
	                                            std::uint32_t timeCost,
	                                            std::uint32_t version,
	                                            std::uint32_t type,
	                                            std::uint32_t lanes)
	{
		(void)keyLength;
		(void)saltBytes;
		(void)outputLength;
		(void)memoryCost;
		(void)timeCost;
		(void)version;
		(void)type;
		(void)lanes;
		return false;
	}

	virtual void run() = 0;
	virtual void runPrepareOnly() { run(); }
	virtual void runHashOnly() { run(); }
	virtual float finish() = 0;
	virtual float getLastHostToDeviceMs() const { return 0.0f; }
	virtual float getLastGpuFirstBlockMs() const { return 0.0f; }
	virtual float getLastDeviceToHostMs() const { return 0.0f; }
	virtual float getLastGpuFinalizeMs() const { return 0.0f; }
};

// Enumerate all available compute devices for the compiled backend.
std::vector<std::unique_ptr<ComputeBackend>> enumerateBackends();
