import time
from contextvars import ContextVar
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field, ValidationError

from nodes.review_runtime import invoke_review
from nodes.aggregator import aggregator_node
from nodes.security_reviewer import security_reviewer_node
from state import OpenCodeReviewState
from state import ChangedFile
from nodes.test_coverage_reviewer import test_coverage_reviewer_node as coverage_node


class Review(BaseModel):
    findings: list = Field(default_factory=list)


def test_tracing_context_survives_deadline_worker():
    trace = ContextVar('test_trace', default='missing')
    token = trace.set('active-trace')
    model = Mock()
    def response(_):
        assert trace.get() == 'active-trace'
        return {'findings': []}
    model.with_structured_output.return_value.invoke.side_effect = response
    try:
        assert invoke_review(model, Review, []).findings == []
    finally:
        trace.reset(token)


def test_fast_documentation_routes_without_test_specialist():
    state = OpenCodeReviewState(diff='+documentation', review_mode='fast', changed_files=[
        ChangedFile(path='README.md', status='modified', content='documentation', diff_hunk='+documentation')])
    assert coverage_node(state)['review_stages'][0]['status'] == 'skipped'


def model_with_results(*results):
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = results
    return model


def test_bare_list_gets_one_repair_not_silent_success():
    model = model_with_results([], {"findings": []})
    result = invoke_review(model, Review, [])
    assert result.findings == []
    assert model.with_structured_output.return_value.invoke.call_count == 2


def test_repeated_invalid_output_fails():
    model = model_with_results([], [])
    with pytest.raises(ValidationError):
        invoke_review(model, Review, [])


def test_provider_error_not_retried():
    model = model_with_results(RuntimeError("429 daily quota"))
    with pytest.raises(RuntimeError):
        invoke_review(model, Review, [])
    assert model.with_structured_output.return_value.invoke.call_count == 1


def test_provider_ignoring_native_timeout_cannot_hold_graph(monkeypatch):
    monkeypatch.setenv("OCR_REVIEWER_TIMEOUT_SECONDS", "0.02")
    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = lambda _: time.sleep(.3)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        invoke_review(model, Review, [])
    assert time.monotonic() - started < .25


def test_failed_review_never_becomes_clean_approval():
    state = OpenCodeReviewState(diff="change", review_stages=[{
        "reviewer": "security", "status": "failed", "error": "ValidationError"}])
    verdict = aggregator_node(state)["verdict"]
    assert verdict.approved is False
    assert verdict.overall_score is None
    assert verdict.recommendation == "incomplete"


def test_unavailable_reviewer_is_explicit_failure(monkeypatch):
    monkeypatch.setattr("nodes.security_reviewer.create_llm", Mock(side_effect=ValueError("Missing key")))
    result = security_reviewer_node(OpenCodeReviewState(diff="change"))
    assert result["review_stages"][0]["status"] == "failed"
    assert result["review_stages"][0]["error"] == "ValueError"
