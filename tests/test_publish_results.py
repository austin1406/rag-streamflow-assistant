import json

import pytest
import requests

from eval import publish_results

RESULTS = [
    {"question": "Q1?", "retrieval_ok": True, "answer_ok": True, "top_sources": ["a"], "answer": "x"},
    {"question": "Q2?", "retrieval_ok": False, "answer_ok": None, "top_sources": [], "answer": None},
]


class FakeResponse:
    def __init__(self, status_code=201, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self._body


def summary(**overrides):
    base = {"id": 3, "label": "baseline", "questions": 2, "retrievalHits": 1, "retrievalRate": 0.5,
            "answersGraded": 1, "answerHits": 1, "answerRate": 1.0}
    return {"summary": {**base, **overrides}}


@pytest.fixture
def results_file(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(RESULTS), encoding="utf-8")
    return path


def test_build_payload_maps_fields_and_drops_extras():
    payload = publish_results.build_payload(RESULTS, "baseline", "llama3.2:1b", 5)

    assert payload == {
        "label": "baseline",
        "model": "llama3.2:1b",
        "topK": 5,
        "results": [
            {"question": "Q1?", "retrievalOk": True, "answerOk": True},
            {"question": "Q2?", "retrievalOk": False, "answerOk": None},
        ],
    }


def test_publish_posts_to_the_runs_endpoint(monkeypatch):
    calls = {}

    def fake_post(url, json, timeout):
        calls.update(url=url, json=json, timeout=timeout)
        return FakeResponse(body=summary())

    monkeypatch.setattr(publish_results.requests, "post", fake_post)

    created = publish_results.publish({"label": "x"}, url="http://host:9000/")

    assert calls["url"] == "http://host:9000/api/runs"
    assert calls["json"] == {"label": "x"}
    assert created["summary"]["id"] == 3


def test_main_stores_a_run_and_reports_rates(monkeypatch, results_file, capsys):
    monkeypatch.setattr(publish_results.requests, "post", lambda url, json, timeout: FakeResponse(body=summary()))

    code = publish_results.main(["--label", "baseline", "--model", "m", "--results", str(results_file)])

    assert code == 0
    out = capsys.readouterr().out
    assert "Stored run 3" in out
    assert "retrieval 1/2 (50%)" in out
    assert "answers 1/1 (100%)" in out


def test_main_reports_ungraded_answers(monkeypatch, results_file, capsys):
    body = summary(answerRate=None, answersGraded=0, answerHits=0)
    monkeypatch.setattr(publish_results.requests, "post", lambda url, json, timeout: FakeResponse(body=body))

    publish_results.main(["--label", "b", "--model", "m", "--results", str(results_file)])

    assert "answers not graded" in capsys.readouterr().out


def test_main_fails_cleanly_when_results_are_missing(tmp_path, capsys):
    code = publish_results.main(["--label", "b", "--model", "m", "--results", str(tmp_path / "nope.json")])

    assert code == 1
    assert "run_eval" in capsys.readouterr().err


def test_main_fails_cleanly_when_the_api_is_down(monkeypatch, results_file, capsys):
    def refuse(url, json, timeout):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(publish_results.requests, "post", refuse)

    code = publish_results.main(["--label", "b", "--model", "m", "--results", str(results_file)])

    assert code == 1
    assert "Could not reach the evaluation API" in capsys.readouterr().err


def test_main_surfaces_api_validation_errors(monkeypatch, results_file, capsys):
    bad = FakeResponse(status_code=400, text='{"errors": {"label": "must not be blank"}}')
    monkeypatch.setattr(publish_results.requests, "post", lambda url, json, timeout: bad)

    code = publish_results.main(["--label", "b", "--model", "m", "--results", str(results_file)])

    assert code == 1
    assert "must not be blank" in capsys.readouterr().err
