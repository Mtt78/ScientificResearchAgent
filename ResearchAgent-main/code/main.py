# WBT evidence extension entrypoint; legacy options continue below.
import sys as _ext_sys
from pathlib import Path as _ExtPath
if __name__ == '__main__' and any(x in _ext_sys.argv[1:] for x in ('--problem', '--paper')):
    _ext_sys.path.insert(0, str(_ExtPath(__file__).resolve().parents[2]))
    from research_extension.cli import main as _evidence_main
    _evidence_main()
    raise SystemExit(0)

import argparse
import json
import os
import time
from pathlib import Path
from tqdm import tqdm
from utils import data_io, s2
from knowledge.store import KnowledgeStore
from models.openai import OpenAIClient
from pipelines.research_pipeline import ResearchPipeline

def fetch_resources(paper, knowledge_store):
    references = s2.get_relevant_references(paper)
    entities = knowledge_store.get_relevant_entities([paper['corpusId']] + [reference['corpusId'] for reference in references])
    return references, entities

def run(paper_ids, knowledge_store, openai_client, iterations=3, output='./results/ideas.jsonl', prepared_context=None):
    if prepared_context is not None:
        contexts = [prepared_context]
    else:
        papers = s2.filter_papers(s2.get_papers(paper_ids), categories=['title', 'abstract', 'embedding'])
        if not papers:
            raise RuntimeError('No usable paper with title, abstract and embedding; refusing an empty run')
        contexts = []
        for paper in papers:
            references, entities = fetch_resources(paper, knowledge_store)
            contexts.append({'paper': {key: paper.get(key) for key in ('paperId', 'corpusId', 'title', 'abstract')}, 'references': references, 'entities': entities})
    results = []
    for context in tqdm(contexts):
        for key in ('paper', 'references', 'entities'):
            if not context.get(key):
                raise ValueError(f'Input is missing a nonempty {key}')
        pipeline = ResearchPipeline(api_client=openai_client, iterations=iterations)
        context = pipeline.run(context)
        data_io.save_result(output, context)
        results.append(context)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', '-d', default='./data/papers.jsonl')
    parser.add_argument('--knowledge-path', '-k', default='./data/knowledge.jsonl')
    parser.add_argument('--model-name', '-m', default='deepseek-r1:8b-0528-qwen3-q4_K_M')
    parser.add_argument('--base-url', default='http://127.0.0.1:11435/v1')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--max-papers', type=int, default=1)
    parser.add_argument('--review-workers', type=int, default=1)
    parser.add_argument('--prepared-context')
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.relative_to(Path(os.environ['WBT_ROOT']).resolve())
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ['RESEARCH_RUN_DIR'] = str(run_dir)
    os.environ['RESEARCH_REVIEW_WORKERS'] = str(args.review_workers)
    prepared = json.loads(Path(args.prepared_context).read_text()) if args.prepared_context else None
    status = {'state': 'running', 'started_at': time.time(), 'config': vars(args), 'gpu_uuid': os.environ.get('CUDA_VISIBLE_DEVICES')}
    (run_dir/'status.json').write_text(json.dumps(status, indent=2))
    try:
        papers = [] if prepared else data_io.load_paper_ids(args.data_path, num_papers=args.max_papers)
        knowledge = None if prepared else KnowledgeStore(args.knowledge_path)
        client = OpenAIClient(model=args.model_name, base_url=args.base_url, log_dir=run_dir)
        results = run(papers, knowledge, client, args.iterations, str(run_dir/'ideas.jsonl'), prepared)
        (run_dir/'result.json').write_text(json.dumps(results[0], ensure_ascii=False, indent=2))
        status.update(state='complete', completed_at=time.time(), paper_count=len(results))
    except Exception as exc:
        status.update(state='failed', completed_at=time.time(), error=repr(exc))
        raise
    finally:
        (run_dir/'status.json').write_text(json.dumps(status, indent=2))
