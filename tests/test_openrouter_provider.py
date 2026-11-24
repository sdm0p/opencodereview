import pytest

from backend.integrations import llm_factory


def test_openrouter_default_and_shared_pacing(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-key')
    model = llm_factory._get_openrouter_chat()
    assert model.model_name == 'openrouter/free'
    assert model.max_retries == 0
    assert model.rate_limiter is llm_factory._openrouter_limiter
    assert llm_factory.create_llm().rate_limiter is model.rate_limiter


def test_daily_budget_rejects_additional_attempts():
    limiter = llm_factory.OpenRouterRateLimiter(daily_limit=2)
    limiter._reserve()
    limiter._reserve()
    with pytest.raises(ValueError, match='daily request budget'):
        limiter._reserve()
