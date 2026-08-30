# x1miner

**XenBlocks GPU miner — 0% dev fee. One fat binary for RTX 30 / 40 / 50 series.**

> Version: v5.1.0-m10000-parked · Linux only (Ubuntu / vast.ai / HiveOS)

Everything you mine stays yours. No fee code exists in this repository — audit it.

## Features

- **0% dev fee** — no fee address, no fee logic, nothing to toggle off
- **Auto-follow difficulty** — mines at the network's current minimum (m=10000+, oscillates with miner count); blocks found at higher diff are parked and flushed automatically when eligible
- **Multi-arch fat binary** — one build runs on Ampere (sm_86), Ada (sm_89) and Blackwell (sm_120a, RTX 5090); CUDA runtime linked statically, so mining rigs don't need the CUDA toolkit
- **Hardware auto-detection** — `build.sh` detects your GPU and picks the right architecture; RTX 5090 gets CUDA 13 handling via `scripts/ensure-cuda13.sh` (installs toolkit only, never touches your driver)
- **VRAM-aware batching** — batch size auto-tuned to the card (e.g. ~955+ on a 12 GB RTX 3060), full ramp-up in minutes
- Mines **XNM / XBLK / XUNI**, exports live state to `data/status.json`

## Quick start (5 minutes, vast.ai / Ubuntu)

```bash
# 1. Get the code
git clone https://github.com/dawidosX/x1miner.git
cd x1miner

# 2. Dependencies
sudo apt-get update && sudo apt-get install -y git build-essential cmake ninja-build pkg-config libcurl4-openssl-dev
# (or: ./install-deps.sh)

# 3. Build & mine — detects your GPU (RTX 5090 -> sm_120a, auto-installs CUDA 13
# compiler if needed), creates miner.ini and asks for your wallet on first run
./start-miner.sh
```

That's it. `start-miner.sh` builds on first run, prompts for your EVM wallet
(0x + 40 hex) and starts mining. Prefer to configure manually? Copy
`miner.ini.example` to `miner.ini` and set `[account] address` before starting.


## Multi-GPU / fleet

One instance = one GPU. Use per-GPU configs with `device_id = N` and a systemd
template unit, or build once on any CUDA machine with `./build-fat-binary.sh`
and copy the single binary to rigs (HiveOS has no nvcc — the static cudart
binary just runs). See HOWTO.md section 4.

## Expected performance (memory-bandwidth-bound)

| GPU | ~kH/s |
|-----|-------|
| RTX 3060 | ~13 |
| RTX 4070 | ~23 |
| RTX 5090 | ~56 |

The algorithm (Argon2, m=10000) is memory-bound: hashrate scales with memory
bandwidth, not core clock. Modest memory OC may help; core OC won't.

## License

MIT — see [LICENSE](LICENSE).

Kernel lineage is shared with the wider XenBlocks open-source ecosystem.
