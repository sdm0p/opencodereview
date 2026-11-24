import pytest

from backend.integrations import llm_factory
from eval_data.ragas_eval import ragas_time_limits


def test_cloudflare_default_and_shared_pacing(monkeypatch):
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'test-key')
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'test-account')
    model = llm_factory._get_cloudflare_chat()
    assert model.model_name == '@cf/zai-org/glm-4.7-flash'
    assert str(model.openai_api_base) == 'https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1'
    assert model.max_retries == 0
    assert model.max_tokens >= 4096
    assert model.reasoning_effort == 'low'
    assert model.rate_limiter is llm_factory._cloudflare_limiter
    assert llm_factory.create_llm().rate_limiter is model.rate_limiter


def test_daily_budget_rejects_additional_attempts():
    limiter = llm_factory.CloudflareRateLimiter(daily_limit=2)
    limiter._reserve()
    limiter._reserve()
    with pytest.raises(ValueError, match='daily request budget'):
        limiter._reserve()


def test_cloudflare_evaluation_has_room_for_paced_calls(monkeypatch):
    monkeypatch.setenv('OCR_RAGAS_TIMEOUT_SECONDS', '30')
    monkeypatch.setenv('OCR_RAGAS_METRIC_TIMEOUT_SECONDS', '12')
    assert ragas_time_limits('Cloudflare') == (210, 150)
    assert ragas_time_limits('Gemini') == (30, 12)
