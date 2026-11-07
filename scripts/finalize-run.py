import datetime,hashlib,json,os,subprocess,sys,urllib.request
from pathlib import Path
R=Path(os.environ['WBT_ROOT']).resolve()
run=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else R/'runs/up5-one-round'
run.relative_to(R)
status=json.loads((run/'status.json').read_text())
assert status['state']=='complete',status
result=json.loads((run/'result.json').read_text())
input_context=json.loads((run/'input_context.json').read_text())
assert result['paper']==input_context['paper']
jsonl=[json.loads(line) for line in (run/'ideas.jsonl').read_text().splitlines() if line]
assert len(jsonl)==1 and jsonl[0]==result
assert result['references']==input_context['references'] and len(result['references'])==10
assert result['entities']==input_context['entities'] and len(result['entities'])==30
for item in [result['paper']]+result['references']:
    assert all(item.get(k) for k in ['paperId','corpusId','title','abstract'])
stages=[('problem','problems','研究问题'),('method','methods','研究方法'),('experiment','experiments','实验设计')]
for stage,plural,label in stages:
    assert result.get(stage) and result.get(stage+'_rationale')
    feedbacks=result[stage+'_feedbacks']
    assert len(feedbacks)==5
    assert all(f.get('review') and f.get('feedback') and type(f.get('rating')) is int and 1<=f['rating']<=5 for f in feedbacks.values())
    assert len(result['history'][plural])==1
    assert result['history'][plural][0][stage]==result[stage]
    assert result['history'][plural][0]['rationale']==result[stage+'_rationale']
    assert result['history'][plural][0]['feedbacks']==feedbacks
records=[json.loads(line) for line in (run/'stage_records.jsonl').read_text().splitlines() if line]
assert {(d['stage'],d['kind']) for d in records}=={(stage,kind) for stage,_,_ in stages for kind in ['generation','feedback']}
calls=[json.loads(line) for line in (run/'llm_calls.jsonl').read_text().splitlines() if line]
assert len({d['call_id'] for d in calls if d['status']=='ok'})>=18
review_calls=[d for d in calls if d.get('response_format') and d['status']=='ok']
assert len(review_calls)>=15
for d in review_calls:
    raw=json.loads(d['response']['choices'][0]['message']['content'])
    assert raw['review'] and raw['feedback'] and 1<=raw['rating']<=5
hashes=json.loads((run/'source-hashes.json').read_text())
assert all(hashlib.sha256((R/'ResearchAgent-main'/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
with urllib.request.urlopen('http://127.0.0.1:11435/api/ps',timeout=10) as f:ps=json.load(f)
with urllib.request.urlopen('http://127.0.0.1:11435/api/version',timeout=10) as f:version=json.load(f)
assert any(d['name']=='deepseek-r1:8b-0528-qwen3-q4_K_M' and d.get('size_vram',0)>0 for d in ps['models'])
configured_gpu=os.environ['CUDA_VISIBLE_DEVICES']
assert configured_gpu==status['gpu_uuid'] and ',' not in configured_gpu
gpu_index=int(subprocess.check_output(['nvidia-smi','-i',configured_gpu,'--query-gpu=index','--format=csv,noheader'],text=True).strip())
gpu=subprocess.check_output(['nvidia-smi','-i',configured_gpu,'--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
assert os.environ['CUDA_VISIBLE_DEVICES'] in gpu
(run/'gpu-evidence.json').write_text(json.dumps({'configured_gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'gpu_index':gpu_index,'ollama_version':version,'loaded_models':ps,'gpu3_processes':gpu},indent=2))
summary={'validation':'passed','paper_title':result['paper']['title'],'paper_corpus_id':result['paper']['corpusId'],'iterations_per_stage':1,'reference_count':len(result['references']),'entity_count':len(result['entities']),'validated_stage_count':len(records),'complete_feedback_count':15,'llm_call_count':len(calls),'unique_llm_calls':len({d['call_id'] for d in calls}),'transport_or_length_retries':sum(d['status']!='ok' for d in calls),'format_retried_stages':sum(d['attempt']>1 for d in records),'elapsed_seconds':status['completed_at']-status['started_at'],'gpu_index':gpu_index,'gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'ollama_version':version['version'],'model':'deepseek-r1:8b-0528-qwen3-q4_K_M','model_weight_sha256':'e6a7edc1a4d7d9b2de136a221a57336b76316cfe53a252aeba814496c5ae439d','stage_ratings':{stage:{metric:f['rating'] for metric,f in result[stage+'_feedbacks'].items()} for stage,_,_ in stages}}
(run/'checkpoint.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for name in hashes:
    source=R/'ResearchAgent-main'/name
    target=run/'source_snapshot'/name
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
(run/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
def section(title,text):return ['## '+title,'',str(text),'']
lines=['# ResearchAgent 单篇单轮复现记录','',f"论文：{result['paper']['title']}",'',f"Corpus ID：{result['paper']['corpusId']}",'',f"模型：`{summary['model']}`；Ollama `{version['version']}`；物理 GPU {gpu_index}。",'',f"三个阶段各运行 1 轮，完成 {len(records)} 个阶段记录、15 项评审，模型调用 {len(calls)} 次，流程耗时 {summary['elapsed_seconds']/60:.2f} 分钟。",'','本记录证明本地模型下的流程与字段完整性；模型自评分不代表独立质量评价或原论文指标复现。','']
lines+=section('目标论文摘要',result['paper']['abstract'])
for stage,plural,label in stages:
    lines+=section(label,result[stage])
    lines+=section(label+'的理由',result[stage+'_rationale'])
    lines+=['## '+label+'：五项评审','']
    for metric,f in result[stage+'_feedbacks'].items():
        lines+=['### '+metric+' — '+str(f['rating'])+'/5','','**Review**', '',f['review'],'','**Feedback**','',f['feedback'],'']
lines+=section('知识实体',' | '.join(result['entities']))
lines+=['## 参考文献与摘要','']
for i,ref in enumerate(result['references'],1):
    score=result['provenance']['selected_similarity_scores'][ref['paperId']]
    lines += [f"### {i}. {ref['title']}",'',f"Corpus ID：{ref['corpusId']}；Paper ID：`{ref['paperId']}`；SPECTER 余弦相似度：{score:.6f}",'',ref['abstract'],'']
lines+=['## 输入与运行记录','', '- `result.json`：全部结果字段和三个阶段的完整 history。','- `ideas.jsonl`：同一结果的 JSONL 格式。','- `input_context.json`：真实论文、参考文献、知识实体与检索来源。','- `stage_records.jsonl`：六个通过校验的阶段输出。','- `llm_calls.jsonl`：完整请求、原始响应、参数、耗时与重试。','- `gpu-evidence.json`：所选 GPU 与模型显存占用证据。','- `requirements-lock.txt`、`source-hashes.json`：依赖版本和运行代码摘要。','- `validation.json`：机器可读完整性检查结果。','','## 重跑','', '```bash','source /home/think/mnt/cyh/610/wbt/activate.sh','/home/think/mnt/cyh/610/wbt/scripts/reproduce-one.sh /home/think/mnt/cyh/610/wbt/runs/up5-one-round-repeat','```','']
(run/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('REPORT',run/'report.md')
