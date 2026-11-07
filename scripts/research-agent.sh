#!/usr/bin/env bash
set -euo pipefail
source /home/think/mnt/cyh/610/wbt/activate.sh
export PYTHONPATH="$WBT_ROOT"
exec "$WBT_ROOT/envs/wbt/bin/python" -m research_extension.cli "$@"
