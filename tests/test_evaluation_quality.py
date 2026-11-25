import sys
import time
from types import SimpleNamespace

import pytest

from eval_data import ragas_eval
from observability import TokenCostCallback, estimate_model_cost
from eval_data.offline_regression import evaluate


def prepare(monkeypatch):
    monkeypatch.setattr(ragas_eval, "_check_ragas", lambda: True)
    monkeypatch.setattr(ragas_eval, "_get_evaluator_llm", lambda endpoint, **kwargs: object())
    monkeypatch.setitem(sys.modules, "ragas.dataset_schema", SimpleNamespace(SingleTurnSample=lambda **kwargs: kwargs))


def test_metrics_keep_success_when_other_metric_times_out(monkeypatch):
    prepare(monkeypatch)
    def score(name, *args):
        if name == "faithfulness":
            time.sleep(.2)
        return .75
    monkeypatch.setattr(ragas_eval, "_score_one_metric", score)
    report = ragas_eval.compute_ragas_report("query", ["context"], answer="answer", metric_timeout=.03)
    assert report["scores"] == {"context_precision": .75}
    assert report["metrics"]["faithfulness"]["status"] == "timed_out"
    assert report["metrics"]["answer_relevancy"]["status"] == "skipped"


def test_failed_judges_never_generate_fake_fallback_scores(monkeypatch):
    prepare(monkeypatch)
    def fail(*args):
        raise RuntimeError("provider unavailable")
    monkeypatch.setattr(ragas_eval, "_score_one_metric", fail)
    report = ragas_eval.compute_ragas_report("same text", ["same text"], answer="same text")
    assert report["scores"] == {}
    assert report["metrics"]["context_precision"]["value"] is None


def test_no_context_is_not_a_zero_quality_measurement():
    report = ragas_eval.compute_ragas_report("query", [], answer="answer")
    assert report["scores"] == {}
    assert all(metric["status"] == "skipped" for metric in report["metrics"].values())


def test_parallel_metrics_have_independent_evaluators(monkeypatch):
    prepare(monkeypatch)
    evaluators = []
    def factory(endpoint, **kwargs):
        evaluator = object()
        evaluators.append(evaluator)
        return evaluator
    monkeypatch.setattr(ragas_eval, "_get_evaluator_llm", factory)
    monkeypatch.setattr(ragas_eval, "_score_one_metric", lambda *args: .5)
    report = ragas_eval.compute_ragas_report("query", ["context"], answer="answer")
    assert len(evaluators) == 2
    assert evaluators[0] is not evaluators[1]
    assert len(report["scores"]) == 2


def test_metric_deadline_is_passed_to_evaluator(monkeypatch):
    prepare(monkeypatch)
    timeouts = []
    monkeypatch.setattr(
        ragas_eval,
        "_get_evaluator_llm",
        lambda endpoint, *, timeout: timeouts.append(timeout) or object(),
    )
    monkeypatch.setattr(ragas_eval, "_score_one_metric", lambda *args: .5)
    report = ragas_eval.compute_ragas_report(
        "query", ["context"], answer="answer", metric_timeout=3
    )
    assert timeouts == [3, 3]
    assert len(report["scores"]) == 2


def test_evaluator_wrapper_uses_supplied_timeout(monkeypatch):
    captured = {}

    class FakeRunConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeWrapper:
        def __init__(self, model, *, run_config):
            captured["model"] = model
            captured["run_config"] = run_config

    monkeypatch.setitem(sys.modules, "ragas.llms", SimpleNamespace(LangchainLLMWrapper=FakeWrapper))
    monkeypatch.setitem(sys.modules, "ragas.run_config", SimpleNamespace(RunConfig=FakeRunConfig))
    ragas_eval._wrap_evaluator(object(), timeout=27)
    assert captured["timeout"] == 27
    assert captured["max_retries"] == 0


def test_metric_errors_are_actionable_without_provider_secrets():
    connection_error = type("OpenAIConnectionError", (Exception,), {})
    status, error = ragas_eval._safe_metric_error(connection_error("secret-api-key"))
    assert status == "failed"
    assert "connection failed" in error
    assert "secret-api-key" not in error
    assert ragas_eval._safe_metric_error(TimeoutError())[0] == "timed_out"


def test_cost_free_estimate_is_distinct_from_provider_report():
    callback = TokenCostCallback()
    callback.on_llm_end(SimpleNamespace(llm_output={"model_name": "minimax/test:free", "token_usage": {"prompt_tokens": 1000, "completion_tokens": 2000}}))
    assert callback.details()["amount"] == 0
    assert callback.details()["source"] == "free_model_estimate"
    assert "not billing" in callback.summary()
    callback.reset()
    callback.on_llm_end(SimpleNamespace(llm_output={"model_name": "unknown", "token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": .123}}))
    assert callback.details()["source"] == "provider_reported"
    assert callback.details()["amount"] == .123


def test_unknown_model_cost_stays_unknown():
    assert estimate_model_cost("unknown", 10000, 10000) is None
    callback = TokenCostCallback()
    callback.on_llm_end(SimpleNamespace(llm_output={"model_name": "unknown", "token_usage": {"prompt_tokens": 10, "completion_tokens": 20}}))
    assert callback.details()["amount"] is None
    assert "unknown" in callback.summary().lower()


def test_offline_matching_counts_duplicate_only_once():
    finding = {"file_path": "auth.ts", "line_start": 10, "category": "security", "comment": "Concurrent race allows reuse"}
    fixtures = [{"label_scope": "exhaustive", "repo": "test/repo", "pr_number": 1, "findings": [{**finding, "concept_terms": ["race", "reuse"]}]}]
    report = evaluate([{"repo": "test/repo", "pr_number": 1, "findings": [finding, finding]}], fixtures)
    assert report["totals"]["matched"] == 1
    assert report["totals"]["duplicate_predictions"] == 1
    assert report["precision"] == .5
    assert report["recall"] == 1
