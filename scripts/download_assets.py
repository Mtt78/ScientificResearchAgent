import concurrent.futures,hashlib,json,os,socket,threading,time,urllib.request
from pathlib import Path
R=Path(os.environ['WBT_ROOT'])
CHUNK=4*1024*1024
lock=threading.Lock()
status={}
original_getaddrinfo=socket.getaddrinfo
counter=0
def resolver(host,port,*args,**kwargs):
    global counter
    if host=='release-assets.githubusercontent.com':
        with lock:
            counter+=1
            ip=['185.199.108.133','185.199.110.133'][counter%2]
        return original_getaddrinfo(ip,port,*args,**kwargs)
    return original_getaddrinfo(host,port,*args,**kwargs)
socket.getaddrinfo=resolver
def save_status():
    with lock:data=dict(status)
    p=R/'state/download-progress.json.tmp'
    p.write_text(json.dumps({'updated':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'downloads':data},indent=2))
    p.replace(R/'state/download-progress.json')
def get_small(url):
    for attempt in range(10):
        try:
            with urllib.request.urlopen(url,timeout=20) as f:return f.read()
        except Exception:
            if attempt==9:raise
            time.sleep(2)
def verify(path,digest):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    if h.hexdigest()!=digest:raise RuntimeError('SHA256 mismatch: '+str(path))
def download(name,url,total,digest,dest,workers=32):
    if name=='ollama' and (R/'state/ollama-relay.enabled').exists():
        while True:
            p=R/'state/ollama-relay-progress.json'
            if p.exists():
                try:
                    data=json.loads(p.read_text())
                    with lock:status[name]=data
                    if data.get('state')=='verified':
                        verify(dest,digest)
                        return
                except (ValueError,OSError):pass
            time.sleep(10)
    parts=R/'downloads'/('parts-'+name);parts.mkdir(exist_ok=True)
    if dest.exists() and dest.stat().st_size==total:
        verify(dest,digest)
        with lock:status[name]={'bytes':total,'total':total,'state':'verified'}
        return
    count=(total+CHUNK-1)//CHUNK
    def part(i):
        start=i*CHUNK;end=min(total,start+CHUNK);path=parts/f'{i:05d}.part'
        for attempt in range(40):
            have=path.stat().st_size if path.exists() else 0
            if have==end-start:return
            if have>end-start:raise RuntimeError('Oversized chunk: '+str(path))
            request=urllib.request.Request(url,headers={'Range':f'bytes={start+have}-{end-1}','User-Agent':'wbt-model-setup/1.0','Accept-Encoding':'identity'})
            try:
                with urllib.request.urlopen(request,timeout=20) as response:
                    expected=f'bytes {start+have}-{end-1}/{total}'
                    if response.status!=206 or response.headers.get('Content-Range')!=expected:
                        raise RuntimeError('Unexpected Range response: '+str(response.headers))
                    with path.open('ab',buffering=0) as out:
                        remaining=end-start-have
                        while remaining:
                            block=response.read(min(65536,remaining))
                            if not block:raise RuntimeError('Truncated HTTP response')
                            out.write(block);remaining-=len(block)
                return
            except Exception as exc:
                print(name,i,'retry',attempt+1,type(exc).__name__,str(exc)[:120],flush=True)
                time.sleep(min(10,1+attempt))
        raise RuntimeError(f'Chunk failed: {name}/{i}')
    started=time.monotonic();initial=sum(p.stat().st_size for p in parts.glob('*.part'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(part,i) for i in range(count)]
        while True:
            done=all(f.done() for f in pending)
            size=sum(p.stat().st_size for p in parts.glob('*.part'))
            with lock:status[name]={'bytes':size,'total':total,'state':'downloading','bytes_per_second':round((size-initial)/max(1,time.monotonic()-started))}
            if done:break
            time.sleep(10)
        for f in pending:f.result()
    with lock:status[name]={'bytes':total,'total':total,'state':'assembling_and_verifying'}
    temp=dest.with_name(dest.name+'.assembling')
    h=hashlib.sha256()
    with temp.open('wb') as out:
        for i in range(count):
            with (parts/f'{i:05d}.part').open('rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''):
                    out.write(block);h.update(block)
    if h.hexdigest()!=digest:raise RuntimeError('SHA256 mismatch: '+name)
    temp.replace(dest)
    for i in range(count):(parts/f'{i:05d}.part').unlink()
    with lock:status[name]={'bytes':total,'total':total,'state':'verified','sha256':digest}
    print(name,'verified',total,digest,flush=True)
registry='https://registry.ollama.ai/v2/library/deepseek-r1/'
tag='8b-0528-qwen3-q4_K_M'
raw=get_small(registry+'manifests/'+tag)
assert hashlib.sha256(raw).hexdigest()=='6995872bfe4c521a67b32da386cd21d5c6e819b6e0d62f79f64ec83be99f5763'
manifest=json.loads(raw)
(R/'downloads/deepseek-manifest.json').write_bytes(raw)
asset=json.loads((R/'downloads/ollama-release.json').read_text())['asset']
blobs=R/'models/ollama/blobs';blobs.mkdir(parents=True,exist_ok=True)
model=next(layer for layer in manifest['layers'] if layer['mediaType']=='application/vnd.ollama.image.model')
errors=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(download,'ollama',asset['browser_download_url'],asset['size'],asset['digest'].split(':')[1],R/'downloads/ollama-linux-amd64.tar.zst'),pool.submit(download,'deepseek',registry+'blobs/'+model['digest'],model['size'],model['digest'].split(':')[1],blobs/model['digest'].replace(':','-'))]
    while not all(f.done() for f in futures):
        save_status();time.sleep(10)
    for f in futures:
        try:f.result()
        except Exception as exc:errors.append(repr(exc))
if errors:raise RuntimeError(str(errors))
for layer in [manifest['config']]+manifest['layers']:
    if layer['digest']==model['digest']:continue
    data=get_small(registry+'blobs/'+layer['digest'])
    assert len(data)==layer['size']
    assert hashlib.sha256(data).hexdigest()==layer['digest'].split(':')[1]
    (blobs/layer['digest'].replace(':','-')).write_bytes(data)
path=R/'models/ollama/manifests/registry.ollama.ai/library/deepseek-r1'/tag
path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
save_status()
print('ALL_DOWNLOADS_VERIFIED',flush=True)
