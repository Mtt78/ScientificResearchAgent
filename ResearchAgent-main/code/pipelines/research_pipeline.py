import json
import os
from pathlib import Path
from typing import Dict
from tqdm.auto import tqdm 

from pipelines.agents import (
    ProblemIdentifier,
    ProblemValidator,
    MethodDeveloper,
    MethodValidator,
    ExperimentDesigner,
    ExperimentValidator,
)
from utils.evaluation import get_avg_feedbacks_score, get_num_feedbacks_scores


class ResearchPipeline:
    def __init__(self, api_client=None, iterations: int = 3):
        self.iterations = max(1, iterations)
        self.problem_identifier = ProblemIdentifier(api_client)
        self.problem_validator = ProblemValidator(api_client)
        self.method_developer = MethodDeveloper(api_client)
        self.method_validator = MethodValidator(api_client)
        self.experiment_designer = ExperimentDesigner(api_client)
        self.experiment_validator = ExperimentValidator(api_client)

        self._labels = [
            'Problem Identifier', 'Problem Validator',
            'Method Developer', 'Method Validator',
            'Experiment Designer', 'Experiment Validator',
        ]
        self._label_width = max(len(s) for s in self._labels)

    def _log(self, message: str) -> None:
        tqdm.write(str(message))

    def _fmt(self, label: str) -> str:
        return f'[{label.ljust(self._label_width)}]'

    def run(self, context: Dict) -> Dict:
        history = {'problems': [], 'methods': [], 'experiments': []}
        context['history'] = history
        self._log(f"{self._fmt('Paper Title')} {context['paper']['title']}")

        # Problem ideation and validation
        for i in range(self.iterations):
            self._log(f"{self._fmt('Problem Identifier')} Iteration {i + 1}/{self.iterations} — generating problem…")
            self._run_agent(self.problem_identifier, context, 'problem', 'generation')
            
            self._log(f"{self._fmt('Problem Validator')} Iteration {i + 1}/{self.iterations} — validating problem…")
            self._run_agent(self.problem_validator, context, 'problem', 'feedback')
            
            history['problems'].append(
                {
                    'problem': context.get('problem'),
                    'rationale': context.get('problem_rationale'),
                    'feedbacks': context.get('problem_feedbacks'),
                }
            )

        # Select the best-scored problem to use for context
        best_problem = max(
            history['problems'], 
            key=lambda problem: get_avg_feedbacks_score(problem.get('feedbacks') or {}) 
            if get_num_feedbacks_scores(problem.get('feedbacks') or {}) > 0 
            else -1
        )
        context.update(
            problem=best_problem.get('problem'),
            problem_rationale=best_problem.get('rationale'),
            problem_feedbacks=best_problem.get('feedbacks'),
        )

        # Method development and validation
        for i in range(self.iterations):
            self._log(f"{self._fmt('Method Developer')} Iteration {i + 1}/{self.iterations} — proposing method…")
            self._run_agent(self.method_developer, context, 'method', 'generation')
            
            self._log(f"{self._fmt('Method Validator')} Iteration {i + 1}/{self.iterations} — validating method…")
            self._run_agent(self.method_validator, context, 'method', 'feedback')
            
            history['methods'].append(
                {
                    'method': context.get('method'),
                    'rationale': context.get('method_rationale'),
                    'feedbacks': context.get('method_feedbacks'),
                }
            )

        # Select the best-scored method to use for context
        best_method = max(
            history['methods'],
            key=lambda method: get_avg_feedbacks_score(method.get('feedbacks') or {})
            if get_num_feedbacks_scores(method.get('feedbacks') or {}) > 0
            else -1,
        )
        context.update(
            method=best_method.get('method'),
            method_rationale=best_method.get('rationale'),
            method_feedbacks=best_method.get('feedbacks'),
        )

        # Experiment design and validation
        for i in range(self.iterations):
            self._log(f"{self._fmt('Experiment Designer')} Iteration {i + 1}/{self.iterations} — designing experiment…")
            self._run_agent(self.experiment_designer, context, 'experiment', 'generation')
            
            self._log(f"{self._fmt('Experiment Validator')} Iteration {i + 1}/{self.iterations} — validating experiment…")
            self._run_agent(self.experiment_validator, context, 'experiment', 'feedback')
            
            history['experiments'].append(
                {
                    'experiment': context.get('experiment'),
                    'rationale': context.get('experiment_rationale'),
                    'feedbacks': context.get('experiment_feedbacks'),
                }
            )

        # Select the best-scored experiment to use for context
        best_experiment = max(
            history['experiments'],
            key=lambda experiment: get_avg_feedbacks_score(experiment.get('feedbacks') or {})
            if get_num_feedbacks_scores(experiment.get('feedbacks') or {}) > 0
            else -1,
        )
        context.update(
            experiment=best_experiment.get('experiment'),
            experiment_rationale=best_experiment.get('rationale'),
            experiment_feedbacks=best_experiment.get('feedbacks'),
        )

        context.update({'history': history})
        return context

    def _run_agent(self, agent, context, stage, kind):
        for attempt in range(2):
            result = agent.run(context)
            if kind == 'generation':
                valid = bool(result.get(stage)) and bool(result.get(stage + '_rationale'))
            else:
                feedbacks = result.get(stage + '_feedbacks', {})
                valid = len(feedbacks) == 5 and all(item.get('review') and item.get('feedback') and isinstance(item.get('rating'), int) and 1 <= item['rating'] <= 5 for item in feedbacks.values())
            if valid:
                context.update(result)
                run_dir = os.getenv('RESEARCH_RUN_DIR')
                if run_dir:
                    root = Path(run_dir)
                    with (root/'stage_records.jsonl').open('a', encoding='utf-8') as out:
                        out.write(json.dumps({'stage': stage, 'kind': kind, 'attempt': attempt + 1, 'output': result}, ensure_ascii=False) + '\n')
                    temp = root/'checkpoint.json.tmp'
                    temp.write_text(json.dumps(context, ensure_ascii=False, indent=2))
                    temp.replace(root/'checkpoint.json')
                return
            self._log(f'[{stage}] invalid {kind} format, retry {attempt + 1}/2')
        raise RuntimeError(f'{stage} {kind} did not return all required fields')
