import pytest

from rag.llm_backend import BackendError, get_backend


def _clear_provider_env(monkeypatch):
    for var in ("LLM_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_defaults_to_ollama_when_nothing_configured(monkeypatch):
    _clear_provider_env(monkeypatch)
    assert get_backend().name == "ollama"


def test_auto_detects_anthropic_from_api_key(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert get_backend().name == "anthropic"


def test_explicit_provider_overrides_auto_detection(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert get_backend().name == "groq"


def test_explicit_provider_without_key_raises_clear_error(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(BackendError, match="OPENAI_API_KEY"):
        get_backend()


def test_unknown_provider_raises_clear_error(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(BackendError, match="Unknown LLM_PROVIDER"):
        get_backend()


class FakeResponse:
    def __init__(self, status_code=200, body=None, headers=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._body


GROQ_OK = {"choices": [{"message": {"content": " hello "}}]}


@pytest.fixture
def groq(monkeypatch):
    from rag import llm_backend

    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    waits = []
    monkeypatch.setattr(llm_backend, "_sleep", waits.append)
    return llm_backend, waits


def test_rate_limited_requests_wait_for_retry_after_then_succeed(groq, monkeypatch):
    llm_backend, waits = groq
    replies = [FakeResponse(429, headers={"retry-after": "7"}), FakeResponse(200, GROQ_OK)]
    monkeypatch.setattr(llm_backend.requests, "post", lambda *a, **k: replies.pop(0))

    assert get_backend().generate("hi") == "hello"
    assert waits == [7.0]


def test_retry_wait_is_capped_and_defaults_to_backoff(groq, monkeypatch):
    llm_backend, waits = groq
    replies = [
        FakeResponse(429, headers={"retry-after": "9999"}),
        FakeResponse(503),
        FakeResponse(200, GROQ_OK),
    ]
    monkeypatch.setattr(llm_backend.requests, "post", lambda *a, **k: replies.pop(0))

    get_backend().generate("hi")

    assert waits == [llm_backend.MAX_RETRY_WAIT_SECONDS, 4.0]


def test_gives_up_after_the_retry_limit_with_a_clear_error(groq, monkeypatch):
    llm_backend, waits = groq
    calls = []

    def always_limited(*a, **k):
        calls.append(1)
        return FakeResponse(429, text="slow down")

    monkeypatch.setattr(llm_backend.requests, "post", always_limited)

    with pytest.raises(BackendError, match="Groq API error 429"):
        get_backend().generate("hi")
    assert len(calls) == llm_backend.MAX_RETRIES + 1
    assert len(waits) == llm_backend.MAX_RETRIES


def test_client_errors_are_not_retried(groq, monkeypatch):
    llm_backend, waits = groq
    monkeypatch.setattr(llm_backend.requests, "post", lambda *a, **k: FakeResponse(401, text="bad key"))

    with pytest.raises(BackendError, match="401"):
        get_backend().generate("hi")
    assert waits == []


def test_groq_defaults_to_a_model_that_follows_the_agent_protocol(groq, monkeypatch):
    llm_backend, _ = groq
    sent = {}

    def capture(url, **kwargs):
        sent.update(kwargs["json"])
        return FakeResponse(200, GROQ_OK)

    monkeypatch.setattr(llm_backend.requests, "post", capture)
    get_backend().generate("hi")

    assert sent["model"] == "qwen/qwen3.8-27b"
    assert sent["max_tokens"] >= 2048
    assert "reasoning_effort" not in sent


def test_gpt_oss_models_get_a_low_reasoning_effort(groq, monkeypatch):
    llm_backend, _ = groq
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    sent = {}

    def capture(url, **kwargs):
        sent.update(kwargs["json"])
        return FakeResponse(200, GROQ_OK)

    monkeypatch.setattr(llm_backend.requests, "post", capture)
    get_backend().generate("hi")

    assert sent["reasoning_effort"] == "low"


def test_non_reasoning_models_do_not_get_a_reasoning_effort(groq, monkeypatch):
    llm_backend, _ = groq
    monkeypatch.setenv("GROQ_MODEL", "llama-test-model")
    sent = {}
    monkeypatch.setattr(llm_backend.requests, "post", lambda url, **k: (sent.update(k["json"]), FakeResponse(200, GROQ_OK))[1])

    get_backend().generate("hi")

    assert "reasoning_effort" not in sent
