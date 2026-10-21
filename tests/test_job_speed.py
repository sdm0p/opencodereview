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


def test_incomplete_review_never_starts_ragas():
    tree = ast.parse(Path('app.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_ragas_inputs_from_state')
    ns = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'app.py', 'exec'), ns)
    state = SimpleNamespace(values={'verdict': SimpleNamespace(recommendation='incomplete'), 'diff': 'change'})
    assert ns['_ragas_inputs_from_state'](state, 'custom') is None


def test_feedback_is_persisted_for_existing_finding_only():
    ns = scheduler()
    ns['_jobs']['job'] = {'id': 'job', 'owner_id': '1', 'status': 'awaiting_approval', 'findings': [{'comment': 'evidence'}]}
    vote = SimpleNamespace(finding_index=0, label='incorrect')
    assert ns['finding_feedback']('job', vote, SimpleNamespace(owner=1)) == {'feedback': {'0': 'incorrect'}}
    ns['_save_job'].assert_called_once()
    with pytest.raises(HTTPException) as absent:
        ns['finding_feedback']('job', SimpleNamespace(finding_index=1, label='correct'), SimpleNamespace(owner=1))
    assert absent.value.status_code == 422
    with pytest.raises(HTTPException) as foreign:
        ns['finding_feedback']('job', vote, SimpleNamespace(owner=2))
    assert foreign.value.status_code == 404


def test_public_job_never_exposes_evaluator_endpoint_snapshot():
    ns = scheduler()
    job = {'id': 'job', 'status': 'incomplete', 'ragas_inputs': {'endpoint': {'api_key': 'private'}}, 'config': {}}
    assert 'ragas_inputs' not in ns['_public_job'](job)


def scheduler():
    tree = ast.parse(Path('api.py').read_text(encoding='utf-8'))
    names = {'create_review', 'cancel_review', '_public_job', 'get_review', 'decide_review', 'finding_feedback'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    ns = dict(__file__=str(Path('api.py').resolve()), Path=Path, Any=Any, Request=object, ReviewRequest=object,
              Header=lambda *a, **k: None, DecisionRequest=object, FindingFeedbackRequest=object, HTTPException=HTTPException, hashlib=hashlib, json=json,
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
