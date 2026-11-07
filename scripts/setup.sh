#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/think/mnt/cyh/610/wbt
source "$ROOT/activate.sh"
exec >>"$ROOT/logs/setup.log" 2>&1
trap 'code=$?; printf "FAILED exit=%s time=%s\n" "$code" "$(date -Is)" > "$ROOT/state/setup.status"' ERR
printf 'downloading_ollama\n' > "$ROOT/state/setup.status"
python -u "$ROOT/scripts/download_assets.py"
printf 'extracting_ollama\n' > "$ROOT/state/setup.status"
python - <<'PYCODE'
import os, subprocess
root=os.environ['WBT_ROOT']
path=root+'/downloads/ollama-linux-amd64.tar.zst'
names=subprocess.check_output(['tar','--zstd','-tf',path],text=True).splitlines()
for name in names:
    if name.startswith('/') or '..' in name.split('/'):
        raise RuntimeError('Unsafe archive path: '+name)
print('Archive member paths checked:',len(names), flush=True)
PYCODE
tar --zstd -xf "$ROOT/downloads/ollama-linux-amd64.tar.zst" -C "$ROOT/runtime/ollama"
ollama --version
printf 'starting_ollama\n' > "$ROOT/state/setup.status"
"$ROOT/scripts/start-ollama.sh"
printf 'pulling_model\n' > "$ROOT/state/setup.status"
ollama pull deepseek-r1:8b-0528-qwen3-q4_K_M
ollama show deepseek-r1:8b-0528-qwen3-q4_K_M > "$ROOT/logs/model-info.txt"
ollama list > "$ROOT/logs/model-list.txt"
python - <<'PYCODE'
import json,os,urllib.request
root=os.environ['WBT_ROOT']
req=urllib.request.Request('http://127.0.0.1:11435/api/show', data=json.dumps({'model':'deepseek-r1:8b-0528-qwen3-q4_K_M'}).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=30) as r: info=json.load(r)
assert info['details']['quantization_level']=='Q4_K_M', info['details']
assert info['details']['family']=='qwen3',info['details']
with open(root+'/logs/model-verification.json','w') as f:json.dump(info,f,indent=2)
print('Model verified:',info['details'])
PYCODE
printf 'complete\n' > "$ROOT/state/setup.status"
