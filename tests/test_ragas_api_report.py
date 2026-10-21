import ast
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock


def test_metric_report_survives_top_level_completion():
    job = {'id': 'job', 'ragas_task_id': 'task'}
    metrics = {'faithfulness': {'status': 'completed', 'score': .8},
               'context_precision': {'status': 'failed', 'error': 'Timeout'}}
    task = {'done': True, 'scores': {'faithfulness': .8}, 'metrics': metrics,
            'evaluation': {'engine': 'RAGAS'}}
    tree = ast.parse(Path('api.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_with_ragas')
    ns = {'Any': Any, '_public_job': lambda item: dict(item), '_ragas_lock': threading.Lock(),
          '_lock': threading.Lock(), '_ragas_tasks': {'task': task}, '_jobs': {'job': job}, '_save_job': Mock()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'api.py', 'exec'), ns)
    result = ns['_with_ragas'](job)
    assert result['ragas']['status'] == 'partial'
    assert result['ragas']['metrics'] == metrics
    assert job['ragas_metrics'] == metrics
