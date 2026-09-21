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
