import requests
from itertools import chain

import torch
def cos_sim(a, b):
    # Preserve the original normalized-dot-product cosine calculation on CPU.
    a = torch.as_tensor(a, dtype=torch.float32, device='cpu')
    b = torch.as_tensor(b, dtype=torch.float32, device='cpu')
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    return torch.nn.functional.normalize(a, p=2, dim=1) @ torch.nn.functional.normalize(b, p=2, dim=1).T



def _request(method, url, headers, params, body=None, timeout=60):
    import hashlib
    import json
    import os
    import time
    from pathlib import Path
    root = Path(os.environ['WBT_ROOT']) / 'cache/s2/requests'
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps([method, url, params, body], sort_keys=True).encode()).hexdigest()
    path = root / (key + '.json')
    if path.exists():
        return json.loads(path.read_text())
    actual_headers = dict(headers)
    if os.getenv('S2_API_KEY'):
        actual_headers['x-api-key'] = os.environ['S2_API_KEY']
    for attempt in range(5):
        response = requests.request(method, url, headers=actual_headers, params=params, json=body, timeout=timeout)
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 4:
            time.sleep(min(60, 15 * 2 ** attempt))
            continue
        response.raise_for_status()
        result = response.json()
        if isinstance(result, dict) and ('error' in result or result.get('code') == '429'):
            raise RuntimeError(str(result))
        path.write_text(json.dumps(result))
        return result
    raise RuntimeError('Semantic Scholar request failed')


def get_request(url, headers, params, timeout=60, return_type=None):
    return _request('GET', url, headers, params, timeout=timeout)


def post_request(url, headers, params, json, timeout=60, return_type=None):
    return _request('POST', url, headers, params, body=json, timeout=timeout)


def batched(items: list, batch_size: int):
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def flatten_list(items: list):
    return (
        list(chain.from_iterable(items))
        if items and items[0] is not None else []
    )


def get_papers(
    ids: list, 
    fields: list = ['corpusId', 'title', 'abstract', 'year', 'publicationDate', 'referenceCount', 'citationCount', 'embedding.specter_v2'], 
    batch_size: int = 100
):
    response = [
        post_request(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            headers={},
            params={'fields': ','.join(fields)},
            json={'ids': ids_batched}
        )
        for ids_batched in batched(ids, batch_size)
    ]
    return flatten_list(response)


def filter_papers(
    papers: list,
    categories: list = ['title', 'abstract', 'embedding']
):
    return [
        paper for paper in papers
        if isinstance(paper, dict)
        and all(paper.get(category) is not None for category in categories)
    ]


def get_relevant_references(
    paper: dict, 
    fields: list = ['paperId', 'corpusId', 'isInfluential', 'title', 'abstract', 'year', 'publicationDate'], 
    batch_size: int = 1000, top_k: int = 10
):
    references = [
        get_request(
            f'https://api.semanticscholar.org/graph/v1/paper/{paper["paperId"]}/references',
            headers={},
            params={'fields': ','.join(fields), 'offset': index*batch_size, 'limit': batch_size},
            return_type={'data': []}
        )['data']
        for index in range(0, paper['referenceCount'] // batch_size + 1)
    ]

    references = [
        reference for reference in flatten_list(references)
        if all(reference['citedPaper'][category] is not None for category in ['paperId', 'title', 'abstract'])
    ]

    reference2embedding = get_paper2embedding([reference['citedPaper']['paperId'] for reference in references])    
    similar_reference_indices = torch.topk(
        cos_sim(
            paper['embedding']['vector'],
            list(reference2embedding.values())
        ),
        k = min(top_k, len(reference2embedding))
    ).indices[0].tolist() if len(reference2embedding) > 0 else []
    similar_reference_ids = [list(reference2embedding.keys())[index] for index in similar_reference_indices]
    
    return [
        reference['citedPaper'] for reference in references 
        if reference['citedPaper']['paperId'] in similar_reference_ids
    ]


def get_paper2embedding(ids: list):
    return {
        paper['paperId']: paper['embedding']['vector'] 
        for paper in get_papers(ids, fields=['embedding.specter_v2'])
        if (
            type(paper) == dict and 
            'embedding' in paper.keys() and 
            paper['embedding'] is not None
        )
    }
