import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from app import app  # noqa: E402


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_health_endpoint_reports_ok(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok"}


def test_agent_endpoint_requires_a_question(client):
    res = client.post("/api/agent", json={})
    assert res.status_code == 400
    assert "question" in res.get_json()["error"]


def test_agent_endpoint_returns_structured_output(client, monkeypatch):
    fake = {"question": "q", "answer": "a", "citations": [], "chunks": [], "tool_calls": [],
            "steps": 1, "warnings": [], "error": None}
    monkeypatch.setattr("app.run_agent", lambda question: fake)

    res = client.post("/api/agent", json={"question": "q"})
    assert res.status_code == 200
    assert res.get_json() == fake
