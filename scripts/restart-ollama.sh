#!/usr/bin/env bash
set -euo pipefail
source /home/think/mnt/cyh/610/wbt/activate.sh
python - <<'PYCODE'
import os,signal,socket,time
from pathlib import Path
root=Path(os.environ['WBT_ROOT']);pidfile=root/'state/ollama.pid'
if pidfile.exists():
    pid=int(pidfile.read_text())
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
for _ in range(30):
    with socket.socket() as s:
        try:s.bind(('127.0.0.1',11435));break
        except OSError:time.sleep(1)
else:raise RuntimeError('Project Ollama endpoint did not stop; inspect project logs')
PYCODE
exec "$WBT_ROOT/scripts/start-ollama.sh"
