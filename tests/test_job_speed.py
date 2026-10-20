"""Exercise API scheduling without importing the heavyweight model runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import hashlib
import json
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import HTTPException


def scheduler():
    tree = ast.parse(Path('api.py').read_text(encoding='utf-8'))
    names = {'create_review', 'cancel_review', '_public_job', 'get_review', 'decide_review'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    ns = dict(__file__=str(Path('api.py').resolve()), Path=Path, Any=Any, Request=object, ReviewRequest=object,
              Header=lambda *a, **k: None, DecisionRequest=object, HTTPException=HTTPException, hashlib=hashlib, json=json,
              threading=threading, time=time, uuid=uuid, datetime=datetime, timezone=timezone,
              os=SimpleNamespace(getenv=lambda *a: ''), requests=SimpleNamespace(get=Mock()),
              _jobs={}, _lock=threading.Lock(), _save_job=Mock(), _executor=SimpleNamespace(submit=Mock()),
              _run_job=Mock(), _load_job=Mock(return_value=None), _authorize_request=Mock(), _rate_limit=Mock(),
              _session=lambda request: {'github_id': request.owner}, _with_ragas=lambda job: dict(job))
    ns['requests'].RequestException = RuntimeError
    ns['requests'].get.return_value.json.return_value = {'head': {'sha': 'a'*40}, 'base': {'sha': 'b'*40}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'api.py', 'exec'), ns)
    return ns


def request(model='m1'):
    values = dict(repo='owner/repo', pr_number=1, endpoint='auto', review_mode='thorough',
                  provider='openai', model=model, api_key='secret', base_url='', baseline_sha=None)
    return SimpleNamespace(**values, model_dump=lambda: values)


def test_identical_active_request_reuses_job_but_other_model_does_not():
    ns = scheduler()
    create = ns['create_review']
    first = create(request(), SimpleNamespace(owner=1))
    second = create(request(), SimpleNamespace(owner=1))
    third = create(request('m2'), SimpleNamespace(owner=1))
    assert second['id'] == first['id'] and second['reused']
    assert third['id'] != first['id']
    assert ns['_executor'].submit.call_count == 2


def test_cache_is_scoped_to_owner_and_immutable_commit():
    ns = scheduler()
    create = ns['create_review']
    first = create(request(), SimpleNamespace(owner=1))
    job = ns['_jobs'][first['id']]
    job.update(status='awaiting_approval', started_timestamp=time.time())
    assert create(request(), SimpleNamespace(owner=1))['id'] == first['id']
    assert create(request(), SimpleNamespace(owner=2))['id'] != first['id']
    ns['requests'].get.return_value.json.return_value['head']['sha'] = 'c'*40
    assert create(request(), SimpleNamespace(owner=1))['id'] != first['id']


def test_cancellation_is_owner_scoped_and_terminal():
    ns = scheduler()
    job = ns['create_review'](request(), SimpleNamespace(owner=1))
    with pytest.raises(HTTPException):
        ns['cancel_review'](job['id'], SimpleNamespace(owner=2))
    assert ns['cancel_review'](job['id'], SimpleNamespace(owner=1))['status'] == 'cancelled'
    assert ns['create_review'](request(), SimpleNamespace(owner=1))['id'] != job['id']
def test_evaluation_quota_stops_followup_calls():
    tree = ast.parse(Path('eval_data/ragas_eval.py').read_text(encoding='utf-8'))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == '_ThrottledChatModel')
    ns = {'_RAGAS_INTERVAL': 0, '_ragas_throttle_lock': threading.Lock(), 'time': time}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'ragas_eval.py', 'exec'), ns)
    model = Mock()
    quota = RuntimeError('rate_limit_exceeded')
    quota.status_code = 429
    model.invoke.side_effect = quota
    wrapped = ns['_ThrottledChatModel'](model)
    with pytest.raises(RuntimeError):
        wrapped.invoke('first')
    with pytest.raises(RuntimeError, match='Evaluation stopped'):
        wrapped.invoke('second')
    assert model.invoke.call_count == 1


def test_read_and_decision_require_job_owner():
    ns = scheduler()
    job = ns['create_review'](request(), SimpleNamespace(owner=1))
    with pytest.raises(HTTPException) as error:
        ns['get_review'](job['id'], SimpleNamespace(owner=2))
    assert error.value.status_code == 404
    with pytest.raises(HTTPException) as error:
        ns['decide_review'](job['id'], SimpleNamespace(action='approve'), SimpleNamespace(owner=2))
    assert error.value.status_code == 404


def test_incremental_review_cannot_post_approval():
    ns = scheduler()
    job = ns['create_review'](request(), SimpleNamespace(owner=1))
    ns['_jobs'][job['id']].update(status='awaiting_approval', baseline_sha='f'*40)
    with pytest.raises(HTTPException) as error:
        ns['decide_review'](job['id'], SimpleNamespace(action='approve'), SimpleNamespace(owner=1))
    assert error.value.status_code == 409
