# x1miner — HOWTO (step-by-step install)

> **Version:** v5.1.0-m10000-parked · x1miner
> Linux only (vast.ai / HiveOS / Ubuntu). Mines XenBlocks at the current network difficulty (min m=10000, auto-follow).
> **0% dev fee** — everything you mine stays yours.

---

## 1. Quick start — vast.ai / Ubuntu with a GPU (5 minutes)

```bash
# 1. Get the code
git clone https://github.com/dawidosX/x1miner.git
cd x1miner

# 2. Dependencies
sudo apt-get update && sudo apt-get install -y git build-essential cmake ninja-build pkg-config libcurl4-openssl-dev
# (or: ./install-deps.sh)

# 3. Build & mine — one command
./start-miner.sh
```

`start-miner.sh` detects your GPU and builds on first run (RTX 5090 → sm_120a;
if nvcc is older than 12.8 it auto-installs the CUDA 13 **compiler only** via
`scripts/ensure-cuda13.sh` — your driver is never touched), creates `miner.ini`
and asks for your EVM wallet (0x + 40 hex chars).

Prefer manual config? `cp miner.ini.example miner.ini` and set
`[account] address` (and optionally `worker`) before starting.

The miner auto-sizes the batch to your VRAM, follows network difficulty
(m=10000+), mines XNM/XBLK/XUNI and exports live state to `data/status.json`.

**Multi-GPU:** one instance = one GPU. Copy the directory per card, or use
separate configs with `device_id = N` (see section 4, HiveOS).

---

## 2. Verify after start

| Dashboard field | Expected |
|---|---|
| Network | `m=10000`+ (ONLINE) |
| Speed | ~10-60 kH/s depending on the card (at m=10000; NOT MH/s — that's normal) |
| Blocks | found grows **and accept grows** within ~1 h |
| Temp | mem junction < 80°C (the miner cuts batch on heat by itself) |

JSON state (every ~4 s): `cat data/status.json` — hashrate, accept, found, temp, VRAM, power.

---

## 3. Important: port 80 (`/difficulty`) and match_drain

The miner learns difficulty from `/difficulty` (port 80) with a fallback to
`lastblock` (port 4445). **If port 80 is blocked on your machine** (common on
some hosting networks, incl. vast):

```bash
curl -s --max-time 8 http://xenblocks.io/difficulty   # empty reply / timeout = blocked
```

→ set in `miner.ini`: `match_drain_enabled = false`
(otherwise match-drain may park the GPU based on the fallback oracle —
symptom: GPU at 0-24%, found barely grows).
Port 80 works → keep `true`.

## 4. HiveOS / farm (many cards)

HiveOS has **no nvcc** — you don't build on the rig. Build the binary once on
any machine with CUDA:

```bash
./build-fat-binary.sh    # one binary: sm_86 (30xx) + sm_89 (40xx) + sm_120a (50xx)
```

Copy the binary to the rigs; per card: a separate directory with its own
`miner.ini` (`device_id = N`, `worker = rig-gpuN`) plus a systemd template unit.
`cudart` is linked statically — target machines need only the NVIDIA driver.

## 5. XUNI — time windows and TIMEZONE

XUNI is mined/submitted only in the **:55-:04** window computed from the
machine's **local time**. The server MUST be in the pool's timezone (**UTC**):
`timedatectl` → if not UTC: `sudo timedatectl set-timezone UTC`.
Wrong timezone = every XUNI rejected as "outside of time window".

## 6. Tuning (optional, `[efficiency]` section)

Defaults are desktop-safe. On headless rigs / farms you can give the batch more VRAM:

```ini
[efficiency]
target_vram_pct = 85
desktop_headroom_pct = 4
gpu_thermal_start_scale = 0.90
thermal_batch_step = 50
```

The miner is memory-bound → **memory OC** helps most (core OC doesn't).
Keep mem junction < 80°C.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| GPU at 0-24%, found stalls | port 80 blocked → `match_drain_enabled = false` (section 3) |
| Speed shown in MH/s | legacy mode — check `memory_cost = 10000`, `force_mine_memory_cost = 0` |
| accept not growing after 1 h | check submit logs, is Network ONLINE (not last-good) |
| `not found` at startup | `ldd build/bin/xnminer` → `sudo apt-get install -y libcurl4` |
| OOM | another process holds VRAM (`nvidia-smi`); auto-sizing uses free VRAM |
| all XUNI rejected | timezone ≠ UTC (section 5) |

## 8. Compatibility

- A binary built on Ubuntu 20.04 runs on HiveOS (GLIBC 2.31) and newer.
  `cudart` is linked statically — the target machine needs only the NVIDIA driver, no CUDA toolkit.
- Requirements: Turing GPU (RTX 20 / GTX 16) or newer, ~1 GB RAM, libcurl4.

---

*x1miner · 2026*
