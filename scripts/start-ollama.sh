#!/usr/bin/env bash
set -euo pipefail
source /home/think/mnt/cyh/610/wbt/activate.sh
python - <<'PYCODE'
import os,socket,subprocess,time,urllib.request,json
root=os.environ['WBT_ROOT']
pidpath=root+'/state/ollama.pid'
if os.path.exists(pidpath):
    try:
        pid=int(open(pidpath).read().strip());os.kill(pid,0)
        with urllib.request.urlopen('http://127.0.0.1:11435/api/version',timeout=3) as r:print(r.read().decode())
        raise SystemExit(0)
    except (OSError,ValueError):pass
with socket.socket() as s:
    s.bind(('127.0.0.1',11435))
with open(root+'/logs/ollama.log','ab') as log:
    p=subprocess.Popen([os.environ['OLLAMA_RUNTIME']+'/bin/ollama','serve'],cwd=root,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
with open(pidpath,'w') as f:f.write(str(p.pid)+'\n')
for _ in range(60):
    if p.poll() is not None:raise RuntimeError('Ollama exited; see logs/ollama.log')
    try:
        with urllib.request.urlopen('http://127.0.0.1:11435/api/version',timeout=2) as r:print(r.read().decode())
        break
    except OSError:time.sleep(1)
else:raise RuntimeError('Ollama startup timed out')
PYCODE
