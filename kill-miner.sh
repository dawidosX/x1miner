#!/usr/bin/env bash
# Stop the miner. Queued blocks already on disk stay in data/ for the next start.
set -euo pipefail

pkill -TERM -f '/build/bin/xnminer' 2>/dev/null || true
pkill -TERM -f 'bash .*/start-miner.sh' 2>/dev/null || true
sleep 2
if pgrep -f '/build/bin/xnminer' >/dev/null 2>&1; then
  pkill -KILL -f '/build/bin/xnminer' 2>/dev/null || true
fi
echo "Miner stopped."
