#!/usr/bin/env bash
# Stop, then start again in this folder. Wallet in miner.ini is kept.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
"${ROOT}/kill-miner.sh"
sleep 1
exec "${ROOT}/start-miner.sh" "$@"
