import pytest

from backend.integrations import llm_factory


def test_cloudflare_default_and_shared_pacing(monkeypatch):
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'test-key')
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'test-account')
    model = llm_factory._get_cloudflare_chat()
    assert model.model_name == '@cf/zai-org/glm-4.7-flash'
    assert str(model.openai_api_base) == 'https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1'
    assert model.max_retries == 0
    assert model.rate_limiter is llm_factory._cloudflare_limiter
    assert llm_factory.create_llm().rate_limiter is model.rate_limiter


def test_daily_budget_rejects_additional_attempts():
    limiter = llm_factory.CloudflareRateLimiter(daily_limit=2)
    limiter._reserve()
    limiter._reserve()
    with pytest.raises(ValueError, match='daily request budget'):
        limiter._reserve()
