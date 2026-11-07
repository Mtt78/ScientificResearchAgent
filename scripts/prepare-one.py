import sys
import json,os,time,urllib.request,urllib.error,hashlib
from pathlib import Path
R=Path(os.environ['WBT_ROOT']);CACHE=R/'cache/s2';CACHE.mkdir(parents=True,exist_ok=True)
def batch(ids,fields,name):
    p=CACHE/name
    if p.exists():return json.loads(p.read_text())
    request=urllib.request.Request('https://api.semanticscholar.org/graph/v1/paper/batch?fields='+fields,data=json.dumps({'ids':ids}).encode(),headers={'Content-Type':'application/json'})
    for i in range(7):
        try:
            with urllib.request.urlopen(request,timeout=45) as f:raw=f.read()
            data=json.loads(raw)
            if not isinstance(data,list):raise RuntimeError(str(data))
            p.write_bytes(raw);return data
        except urllib.error.HTTPError as e:
            body=e.read().decode();print('S2_HTTP',name,e.code,body[:300],flush=True)
            if e.code not in [429,500,502,503,504] or i==6:raise
            time.sleep(min(60,15*2**i))
        except OSError as e:
            print('S2_NETWORK',repr(e),flush=True)
            if i==6:raise
            time.sleep(15)
fields='paperId,corpusId,title,abstract,year,referenceCount,embedding.specter_v2,references.paperId,references.corpusId,references.title,references.abstract'
seed=batch(['CorpusId:258832356'],fields,'seed-reference-metadata.json')[0]
print('SEED_KEYS',list(seed),flush=True)
refs=[d for d in seed.get('references',[]) if d and d.get('paperId') and d.get('title') and d.get('abstract')]
if not refs:raise RuntimeError('No complete references in Semantic Scholar response')
print('REFERENCE_CANDIDATES',len(refs),flush=True)
vectors=batch([d['paperId'] for d in refs],'paperId,corpusId,title,abstract,embedding.specter_v2','reference-vectors-258832356.json')
lookup={d['paperId']:d for d in vectors if isinstance(d,dict) and d.get('embedding') and d['embedding'].get('vector')}
if not seed.get('embedding'):
    seed['embedding']=json.loads((CACHE/'seed-batch.json').read_text())[0]['embedding']
# Identical normalized dot product to sentence_transformers.util.cos_sim; CPU only.
import torch
from torch.nn.functional import normalize
from knowledge.store import KnowledgeStore
ids=list(lookup)
a=torch.tensor(seed['embedding']['vector'],dtype=torch.float32).unsqueeze(0)
b=torch.tensor([lookup[k]['embedding']['vector'] for k in ids],dtype=torch.float32)
scores=(normalize(a,p=2,dim=1) @ normalize(b,p=2,dim=1).T)[0]
indices=torch.topk(scores,k=min(10,len(ids))).indices.tolist()
selected={ids[i]:float(scores[i]) for i in indices}
references=[{k:d.get(k) for k in ['paperId','corpusId','title','abstract']} for d in refs if d['paperId'] in selected]
store=KnowledgeStore(str(R/'ResearchAgent-main/data/knowledge.jsonl'))
entities=store.get_relevant_entities([seed['corpusId']]+[d['corpusId'] for d in references])
assert len(references)==10 and entities
context={'paper':{k:seed.get(k) for k in ['paperId','corpusId','title','abstract']},'references':references,'entities':entities,'provenance':{'source':'Semantic Scholar Graph API batch (cached)','target_id':'CorpusId:258832356','reference_candidates_with_abstract':len(refs),'reference_candidates_with_specter_v2':len(lookup),'retrieval':'cosine similarity of Semantic Scholar SPECTER v2 embeddings, top_k=10; CPU float32','selected_similarity_scores':selected,'knowledge_path':str(R/'ResearchAgent-main/data/knowledge.jsonl'),'prepared_at':time.time(),'source_cache_modified_at':{name:(CACHE/name).stat().st_mtime for name in ['seed-reference-metadata.json','reference-vectors-258832356.json']}}}
run=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else R/'runs/up5-one-round'
run.relative_to(R.resolve());run.mkdir(parents=True,exist_ok=True)
(run/'input_context.json').write_text(json.dumps(context,ensure_ascii=False,indent=2))
print('PREPARED_CONTEXT',run/'input_context.json','references',len(references),'entities',len(entities),flush=True)
