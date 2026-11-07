#!/usr/bin/env bash
set -euo pipefail
source /home/think/mnt/cyh/610/wbt/activate.sh
export PYTHONPATH="$WBT_ROOT/ResearchAgent-main/code"
RUN_DIR="${1:-$WBT_ROOT/runs/up5-one-round}"
if [ -e "$RUN_DIR/result.json" ]; then
    echo "Completed result already exists. Pass a new run directory to preserve it." >&2
    exit 1
fi
python "$WBT_ROOT/scripts/prepare-one.py" "$RUN_DIR"
python "$WBT_ROOT/scripts/snapshot-run.py" "$RUN_DIR"
cd "$WBT_ROOT/ResearchAgent-main"
python -u code/main.py --prepared-context "$RUN_DIR/input_context.json" --iterations 1 --max-papers 1 --review-workers 1 --model-name deepseek-r1:8b-0528-qwen3-q4_K_M --base-url http://127.0.0.1:11435/v1 --run-dir "$RUN_DIR"

python "$WBT_ROOT/scripts/finalize-run.py" "$RUN_DIR"
