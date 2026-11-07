import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse
from openai import OpenAI


def normalize_section_labels(text):
    labels = r'(?:Research[ \t]+|Scientific[ \t]+)?(Problem|Method|Experiment|Rationale|Review|Feedback|Rating(?:[ \t]*\([ \t]*1[-–]5[ \t]*\))?)'
    def canonical(match):
        value = match.group(1)
        return ('Rating (1-5)' if value.lower().startswith('rating') and '(' in value else value.capitalize()) + ':'
    # Standalone Markdown headings may omit the colon requested in the prompt.
    text = re.sub(r'(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?' + labels + r'(?:\*\*|__)?[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*$', canonical, text)
    # Inline labels may be bold, including the colon itself.
    text = re.sub(r'(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?' + labels + r'(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?[ \t]*', lambda m: canonical(m) + ' ', text)
    return text

class OpenAIClient:
    def __init__(self, model='deepseek-r1:8b-0528-qwen3-q4_K_M', base_url=None, log_dir=None):
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'http://127.0.0.1:11435/v1')
        if urlparse(self.base_url).hostname not in ('127.0.0.1', 'localhost'):
            raise ValueError('This workspace reproduction requires a local model endpoint')
        self._client = OpenAI(base_url=self.base_url, api_key='ollama', timeout=float(os.getenv('LLM_TIMEOUT', '600')), max_retries=0)
        self.model = model
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', '8192'))
        self.temperature = float(os.getenv('LLM_TEMPERATURE', '0.6'))
        self.top_p = float(os.getenv('LLM_TOP_P', '0.95'))
        self.seed = int(os.getenv('LLM_SEED', '123'))
        self.log_dir = Path(log_dir or os.getenv('RESEARCH_RUN_DIR', './results'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._call_id = 0

    def call(self, messages: List[Dict[str, str]], max_retries: int = 2) -> str:
        with self._lock:
            self._call_id += 1
            call_id = self._call_id
            token_limit = self.max_tokens
            for attempt in range(max_retries + 1):
                started = time.time()
                record = {'call_id': call_id, 'attempt': attempt + 1, 'started_at': started, 'model': self.model, 'base_url': self.base_url, 'messages': messages, 'max_tokens': token_limit, 'temperature': self.temperature, 'top_p': self.top_p, 'seed': self.seed}
                try:
                    request_messages = [dict(message) for message in messages]
                    last_prompt = request_messages[-1]['content']
                    is_review = all(label in last_prompt for label in ('Review:', 'Feedback:', 'Rating (1-5):'))
                    params = dict(model=self.model, messages=request_messages, max_tokens=token_limit, temperature=self.temperature, top_p=self.top_p, seed=self.seed)
                    if is_review:
                        schema = {'type': 'object', 'properties': {'review': {'type': 'string', 'minLength': 1}, 'feedback': {'type': 'string', 'minLength': 1}, 'rating': {'type': 'integer', 'minimum': 1, 'maximum': 5}}, 'required': ['review', 'feedback', 'rating'], 'additionalProperties': False}
                        params['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'research_review', 'strict': True, 'schema': schema}}
                        request_messages[-1]['content'] += '\n\nSerialization requirement: Return only a JSON object matching the supplied schema. Put the requested review, feedback, and integer rating in their corresponding fields; preserve the requested evaluation criterion.'
                        record['response_format'] = params['response_format']
                    record['messages'] = request_messages
                    response = self._client.chat.completions.create(**params)
                    record['response'] = response.model_dump()
                    choice = response.choices[0]
                    text = choice.message.content or ''
                    if choice.finish_reason == 'length':
                        token_limit = min(16384, token_limit * 2)
                        raise ValueError('Model output was truncated before a complete final answer')
                    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
                    if '<think>' in text and '</think>' not in text:
                        raise ValueError('No complete final answer after thinking')
                    if not text:
                        raise ValueError('Model returned no final answer')
                    if is_review:
                        feedback = json.loads(text)
                        if not (feedback.get('review') and feedback.get('feedback') and type(feedback.get('rating')) is int and 1 <= feedback['rating'] <= 5):
                            raise ValueError('Incomplete structured review')
                        text = 'Review: ' + feedback['review'] + '\nFeedback: ' + feedback['feedback'] + '\nRating (1-5): ' + str(feedback['rating'])
                    # Accept Markdown-decorated section labels while preserving the prompt and answer.
                    text = normalize_section_labels(text)
                    record['final_content'] = text
                    record['status'] = 'ok'
                    return text
                except Exception as exc:
                    record['status'] = 'error'
                    record['error'] = repr(exc)
                    if attempt == max_retries:
                        raise RuntimeError(f'Local LLM call {call_id} failed after {attempt + 1} attempts: {exc}') from exc
                finally:
                    record['elapsed_seconds'] = time.time() - started
                    with (self.log_dir / 'llm_calls.jsonl').open('a', encoding='utf-8') as out:
                        out.write(json.dumps(record, ensure_ascii=False) + '\n')
                    print(f'[LLM] call={call_id} attempt={attempt + 1} status={record["status"]} seconds={record["elapsed_seconds"]:.1f}', flush=True)
                time.sleep(min(30, 4 ** (attempt + 1)))
