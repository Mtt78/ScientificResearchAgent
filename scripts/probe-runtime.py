import os,json,time,urllib.request,urllib.error,subprocess,concurrent.futures
from pathlib import Path
R=Path(os.environ['WBT_ROOT'])
(R/'cache/s2').mkdir(parents=True,exist_ok=True)
def model():
    payload={'model':'deepseek-r1:8b-0528-qwen3-q4_K_M','messages':[{'role':'user','content':'Reply exactly OK.'}],'stream':False,'think':False,'options':{'num_predict':32,'num_ctx':16384,'temperature':0.0},'keep_alive':'30m'}
    def call(p):
        req=urllib.request.Request('http://127.0.0.1:11435/api/chat',data=json.dumps(p).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as f:return json.load(f)
    try:out=call(payload)
    except urllib.error.HTTPError as exc:
        print('MODEL_HTTP_ERROR',exc.code,exc.read().decode(),flush=True)
        if exc.code!=400:raise
        payload.pop('think');out=call(payload)
    (R/'logs/gpu-smoke.json').write_text(json.dumps(out,indent=2))
    print('MODEL_SMOKE',out,flush=True)
    with urllib.request.urlopen('http://127.0.0.1:11435/api/ps',timeout=10) as f:ps=json.load(f)
    (R/'logs/gpu-model-ps.json').write_text(json.dumps(ps,indent=2))
    print('MODEL_PS',ps,flush=True)
    gpu=subprocess.run(['nvidia-smi','-i','3','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
    (R/'logs/gpu3-processes.txt').write_text(gpu.stdout+gpu.stderr)
    print('GPU3_PROCESSES',gpu.stdout,gpu.stderr,flush=True)
def data():
    url='https://api.semanticscholar.org/graph/v1/paper/CorpusId:258832356?fields=paperId,corpusId,title,abstract,year,referenceCount,embedding.specter_v2'
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=25) as f:out=json.load(f)
            (R/'cache/s2/seed-258832356.json').write_text(json.dumps(out,indent=2))
            print('PAPER_FETCH',out.get('title'),'abstract',bool(out.get('abstract')),'embedding',bool(out.get('embedding')),'referenceCount',out.get('referenceCount'),flush=True)
            return
        except Exception as exc:
            print('PAPER_FETCH_ERROR',repr(exc),flush=True);time.sleep(5)
    raise RuntimeError('Unable to fetch seed paper')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(model),pool.submit(data)]
    for f in futures:f.result()
