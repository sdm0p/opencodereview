import pytest
import httpx

from backend.integrations import llm_factory
from eval_data.ragas_eval import ragas_time_limits
from backend.integrations.cloudflare_transport import CloudflareTransport, _routed_request


def test_cloudflare_default_and_shared_pacing(monkeypatch):
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'test-key')
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'a' * 32)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    monkeypatch.delenv('CLOUDFLARE_GATEWAY_ID', raising=False)
    model = llm_factory._get_cloudflare_chat()
    assert model.model_name == '@cf/zai-org/glm-4.7-flash'
    assert str(model.openai_api_base) == f'https://api.cloudflare.com/client/v4/accounts/{"a" * 32}/ai/v1'
    assert model.max_retries == 0
    assert model.max_tokens >= 4096
    assert model.reasoning_effort == 'low'
    assert model.rate_limiter is llm_factory._cloudflare_limiter
    assert llm_factory.create_llm().rate_limiter is model.rate_limiter


def test_automatic_provider_order_is_gemini_then_groq_then_cloudflare(monkeypatch):
    monkeypatch.setattr(llm_factory, '_get_gemini_chat', lambda **_: 'Gemini')
    monkeypatch.setattr(llm_factory, '_get_groq_chat', lambda **_: 'Groq')
    monkeypatch.setattr(llm_factory, '_get_cloudflare_chat', lambda **_: 'Cloudflare')
    assert llm_factory.create_llm() == 'Gemini'

    monkeypatch.setattr(llm_factory, '_get_gemini_chat', lambda **_: None)
    assert llm_factory.create_llm() == 'Groq'

    monkeypatch.setattr(llm_factory, '_get_groq_chat', lambda **_: None)
    assert llm_factory.create_llm() == 'Cloudflare'


def test_gateway_uses_cloudflare_model_and_disables_cache_and_logs(monkeypatch):
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'test-key')
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'a' * 32)
    monkeypatch.setenv('CLOUDFLARE_GATEWAY_ID', 'default')
    model = llm_factory._get_cloudflare_chat()
    assert model.model_name == 'workers-ai/@cf/zai-org/glm-4.7-flash'
    assert str(model.openai_api_base) == f'https://gateway.ai.cloudflare.com/v1/{"a" * 32}/default/compat'
    assert model.default_headers['cf-aig-collect-log'] == 'false'
    assert model.default_headers['cf-aig-skip-cache'] == 'true'
    assert model.rate_limiter is llm_factory._cloudflare_limiter


def test_gateway_supports_a_separate_gateway_token(monkeypatch):
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'workers-ai-token')
    monkeypatch.setenv('CLOUDFLARE_GATEWAY_TOKEN', 'gateway-run-token')
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'a' * 32)
    monkeypatch.setenv('CLOUDFLARE_GATEWAY_ID', 'default')
    model = llm_factory._get_cloudflare_chat()
    assert model.default_headers['cf-aig-authorization'] == 'Bearer gateway-run-token'
    assert model.openai_api_key.get_secret_value() == 'workers-ai-token'


def test_daily_budget_rejects_additional_attempts():
    limiter = llm_factory.CloudflareRateLimiter(daily_limit=2)
    limiter._reserve()
    limiter._reserve()
    with pytest.raises(ValueError, match='daily request budget'):
        limiter._reserve()


def test_cloudflare_evaluation_has_room_for_paced_calls(monkeypatch):
    monkeypatch.setenv('OCR_RAGAS_TIMEOUT_SECONDS', '90')
    monkeypatch.setenv('OCR_RAGAS_METRIC_TIMEOUT_SECONDS', '30')
    assert ragas_time_limits('Cloudflare') == (210, 150)
    assert ragas_time_limits('Gemini') == (90, 30)


def test_routing_keeps_host_and_tls_server_name():
    original = httpx.Request('POST', 'https://api.cloudflare.com/client/v4/accounts/test/ai/v1/chat/completions',
                             headers={'Authorization': 'Bearer test-key'}, content=b'{}')
    routed = _routed_request(original, '192.0.2.1')
    assert routed.url.host == '192.0.2.1'
    assert routed.headers['Host'] == 'api.cloudflare.com'
    assert routed.extensions['sni_hostname'] == 'api.cloudflare.com'
    assert routed.headers['Authorization'] == 'Bearer test-key'
    with pytest.raises(ValueError):
        _routed_request(httpx.Request('POST', 'https://elsewhere.example/'), '192.0.2.1')


def test_gateway_routing_keeps_its_host_and_tls_server_name():
    original = httpx.Request('POST', 'https://gateway.ai.cloudflare.com/v1/acct/default/compat/chat/completions',
                             headers={'Authorization': 'Bearer test-key'}, content=b'{}')
    routed = _routed_request(original, '192.0.2.1', 'gateway.ai.cloudflare.com')
    assert routed.url.host == '192.0.2.1'
    assert routed.headers['Host'] == 'gateway.ai.cloudflare.com'
    assert routed.extensions['sni_hostname'] == 'gateway.ai.cloudflare.com'
    with pytest.raises(ValueError):
        _routed_request(original, '192.0.2.1', 'not-cloudflare.example')


def test_transport_tries_another_route_only_for_connect_failure():
    class Routes:
        def __init__(self):
            self.invalidated = []
        def select(self, excluded):
            return '192.0.2.1' if not excluded else '192.0.2.2'
        def invalidate(self, address):
            self.invalidated.append(address)
    routes = Routes()
    transport = CloudflareTransport(routes)
    def respond(request):
        if request.url.host == '192.0.2.1':
            raise httpx.ConnectError('connection failed')
        return httpx.Response(200, json={'ok': True})
    transport.transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as client:
        response = client.post('https://api.cloudflare.com/example', json={})
    assert response.status_code == 200
    assert routes.invalidated == ['192.0.2.1']


def test_transport_rechecks_when_no_route_is_temporarily_reachable(monkeypatch):
    monkeypatch.setattr('backend.integrations.cloudflare_transport.time.sleep', lambda _: None)
    class Routes:
        calls = 0
        def select(self, excluded):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError('No route')
            return '192.0.2.1'
        def invalidate(self, address):
            raise AssertionError('reachable route should not be invalidated')
    routes = Routes()
    transport = CloudflareTransport(routes)
    transport.transport = httpx.MockTransport(lambda request: httpx.Response(200))
    with httpx.Client(transport=transport) as client:
        assert client.get('https://api.cloudflare.com/example').status_code == 200
    assert routes.calls == 2
