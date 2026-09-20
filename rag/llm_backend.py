"""Pluggable generation backends so the assistant runs on whatever you have: a free local
Ollama model (zero cost) or an API key you already hold (Anthropic / OpenAI / Groq).

Backend is chosen by LLM_PROVIDER env var ("ollama" | "anthropic" | "openai" | "groq"),
or auto-detected from whichever API key is set, falling back to Ollama.
"""
import os
import requests


class BackendError(RuntimeError):
    pass


def _require_key(env_var: str) -> str:
    value = os.getenv(env_var)
    if not value:
        raise BackendError(f"{env_var} is not set.")
    return value


class OllamaBackend:
    name = "ollama"

    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "llama3.1")
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def _unreachable(self):
        return BackendError(
            "Could not reach Ollama at "
            f"{self.host}. Install it from https://ollama.com, run "
            f"`ollama pull {self.model}`, and make sure `ollama serve` is running."
        )

    def health_check(self):
        try:
            requests.get(f"{self.host}/api/tags", timeout=2).raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            r.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise self._unreachable() from e
        return r.json()["response"].strip()


class AnthropicBackend:
    name = "anthropic"

    def __init__(self):
        self.api_key = _require_key("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    def generate(self, prompt: str) -> str:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 800,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        if r.status_code >= 400:
            raise BackendError(f"Anthropic API error {r.status_code}: {r.text}")
        return r.json()["content"][0]["text"].strip()


class OpenAIBackend:
    name = "openai"

    def __init__(self):
        self.api_key = _require_key("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(self, prompt: str) -> str:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
            },
            timeout=120,
        )
        if r.status_code >= 400:
            raise BackendError(f"OpenAI API error {r.status_code}: {r.text}")
        return r.json()["choices"][0]["message"]["content"].strip()


class GroqBackend:
    name = "groq"

    def __init__(self):
        self.api_key = _require_key("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    def generate(self, prompt: str) -> str:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 800,
            },
            timeout=120,
        )
        if r.status_code >= 400:
            raise BackendError(f"Groq API error {r.status_code}: {r.text}")
        return r.json()["choices"][0]["message"]["content"].strip()


_BACKENDS = {
    "ollama": OllamaBackend,
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "groq": GroqBackend,
}


def get_backend():
    provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    if not provider:
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("GROQ_API_KEY"):
            provider = "groq"
        else:
            provider = "ollama"

    if provider not in _BACKENDS:
        raise BackendError(f"Unknown LLM_PROVIDER '{provider}'. Choose from {list(_BACKENDS)}.")
    return _BACKENDS[provider]()
