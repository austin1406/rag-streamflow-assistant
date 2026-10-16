import json

import pytest

from rag import eval_db
from rag.agent import AgentFormatError, parse_message, run_agent
from rag.llm_backend import BackendError


class ScriptedBackend:
    """Replays a fixed list of replies and records the prompts it was given."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def fake_retrieve(question, top_k=5):
    return [
        {"id": "a", "text": "R2 was 0.954", "source": "report.pdf", "loc": "page 2", "distance": 0.1},
        {"id": "b", "text": "coverage was 54.1%", "source": "slides.pptx", "loc": "slide 22", "distance": 0.2},
    ][:top_k]


def tool(name, **args):
    return json.dumps({"tool": name, "args": args})


def final(answer, citations):
    return json.dumps({"final": {"answer": answer, "citations": citations}})


def test_parse_message_accepts_tool_call_inside_extra_text():
    msg = parse_message('Sure! {"tool": "search_documents", "args": {"query": "x"}} done')
    assert msg == {"type": "tool", "name": "search_documents", "args": {"query": "x"}}


@pytest.mark.parametrize(
    "raw",
    [
        "no json here",
        "{not valid json}",
        "[1, 2]",
        '{"neither": 1}',
        '{"tool": "rm_rf", "args": {}}',
        '{"tool": "search_documents", "args": "oops"}',
        '{"final": "just a string"}',
        '{"final": {"answer": "", "citations": []}}',
        '{"final": {"answer": "ok", "citations": ["1"]}}',
        '{"final": {"answer": "ok", "citations": [true]}}',
    ],
)
def test_parse_message_rejects_malformed_replies(raw):
    with pytest.raises(AgentFormatError):
        parse_message(raw)


def test_search_then_final_answer_with_mapped_citations():
    backend = ScriptedBackend(
        [
            tool("search_documents", query="bunchgrass r2", top_k=2),
            final("R2 was 0.954 [1].", [1]),
        ]
    )
    result = run_agent("What was the R2?", backend=backend, retrieve_fn=fake_retrieve)

    assert result["error"] is None
    assert result["answer"] == "R2 was 0.954 [1]."
    assert result["citations"] == [{"n": 1, "source": "report.pdf", "loc": "page 2"}]
    assert result["steps"] == 2
    assert result["tool_calls"] == [
        {"tool": "search_documents", "args": {"query": "bunchgrass r2", "top_k": 2}, "ok": True, "detail": "2 passages"}
    ]
    assert [c["n"] for c in result["chunks"]] == [1, 2]
    assert "[1] (source: report.pdf, page 2)" in backend.prompts[1]


def test_citation_numbers_stay_stable_across_repeated_searches():
    backend = ScriptedBackend(
        [
            tool("search_documents", query="first", top_k=2),
            tool("search_documents", query="second", top_k=2),
            final("done [2]", [2]),
        ]
    )
    result = run_agent("q", backend=backend, retrieve_fn=fake_retrieve)

    assert [c["n"] for c in result["chunks"]] == [1, 2]
    assert result["citations"] == [{"n": 2, "source": "slides.pptx", "loc": "slide 22"}]


def test_invented_citations_are_dropped_and_flagged():
    backend = ScriptedBackend(
        [
            tool("search_documents", query="x", top_k=2),
            final("answer [1] [7]", [1, 7]),
        ]
    )
    result = run_agent("q", backend=backend, retrieve_fn=fake_retrieve)

    assert [c["n"] for c in result["citations"]] == [1]
    assert any("[7]" in w for w in result["warnings"])


def test_answer_with_no_valid_citation_is_flagged():
    backend = ScriptedBackend([tool("search_documents", query="x"), final("no cites", [])])
    result = run_agent("q", backend=backend, retrieve_fn=fake_retrieve)

    assert result["answer"] == "no cites"
    assert any("cites no retrieved passage" in w for w in result["warnings"])


def test_agent_recovers_from_an_invalid_reply():
    backend = ScriptedBackend(["I think the answer is 0.954", final("0.954", [])])
    result = run_agent("q", backend=backend, retrieve_fn=fake_retrieve)

    assert result["answer"] == "0.954"
    assert result["steps"] == 2
    assert any("invalid reply" in w for w in result["warnings"])
    assert "invalid reply" in backend.prompts[1]


def test_bad_tool_arguments_are_reported_back_to_the_model():
    backend = ScriptedBackend(
        [tool("search_documents", query=""), tool("search_documents", query="ok"), final("done [1]", [1])]
    )
    result = run_agent("q", backend=backend, retrieve_fn=fake_retrieve)

    assert result["tool_calls"][0]["ok"] is False
    assert result["tool_calls"][1]["ok"] is True
    assert "query must be a non-empty string" in backend.prompts[1]


def test_agent_stops_at_the_step_limit():
    backend = ScriptedBackend([tool("search_documents", query="x")] * 3)
    result = run_agent("q", backend=backend, retrieve_fn=fake_retrieve, max_steps=3)

    assert result["answer"] is None
    assert result["steps"] == 3
    assert "did not give a final answer within 3 steps" in result["error"]


def test_sql_tool_is_reachable_through_the_agent(tmp_path):
    questions_path = tmp_path / "q.json"
    questions_path.write_text(
        json.dumps([{"question": "Q?", "expect_source_contains": "report", "expect_answer_contains": ["a"]}]),
        encoding="utf-8",
    )
    db_path = tmp_path / "eval.sqlite"
    eval_db.build(db_path=db_path, questions_path=questions_path, retrieve_fn=fake_retrieve)

    backend = ScriptedBackend(
        [
            tool("query_eval_results", sql="SELECT COUNT(*) AS n FROM eval_questions"),
            final("There is 1 evaluation question.", []),
        ]
    )
    result = run_agent("How many eval questions?", backend=backend, db_path=db_path)

    assert result["answer"] == "There is 1 evaluation question."
    assert result["tool_calls"][0]["ok"] is True
    assert '"rows": [[1]]' in backend.prompts[1]
    assert not any("cites no retrieved passage" in w for w in result["warnings"])


def test_destructive_sql_from_the_model_is_blocked(tmp_path):
    questions_path = tmp_path / "q.json"
    questions_path.write_text(
        json.dumps([{"question": "Q?", "expect_source_contains": None, "expect_answer_contains": ["a"]}]),
        encoding="utf-8",
    )
    db_path = tmp_path / "eval.sqlite"
    eval_db.build(db_path=db_path, questions_path=questions_path, retrieve_fn=fake_retrieve)

    backend = ScriptedBackend(
        [tool("query_eval_results", sql="DROP TABLE eval_questions"), final("could not do that", [])]
    )
    result = run_agent("drop it", backend=backend, db_path=db_path)

    assert result["tool_calls"][0]["ok"] is False
    check = ScriptedBackend([tool("query_eval_results", sql="SELECT COUNT(*) FROM eval_questions"), final("ok", [])])
    still_there = run_agent("count", backend=check, db_path=db_path)
    assert still_there["tool_calls"][0]["ok"] is True


def test_backend_errors_are_returned_not_raised():
    class Failing:
        name = "failing"

        def generate(self, prompt):
            raise BackendError("backend down")

    result = run_agent("q", backend=Failing(), retrieve_fn=fake_retrieve)
    assert result["error"] == "backend down"
    assert result["answer"] is None


def test_retrieval_crashes_become_tool_errors():
    def broken(question, top_k=5):
        raise RuntimeError("index missing")

    backend = ScriptedBackend([tool("search_documents", query="x"), final("cannot answer", [])])
    result = run_agent("q", backend=backend, retrieve_fn=broken)

    assert result["tool_calls"][0]["ok"] is False
    assert "index missing" in result["tool_calls"][0]["detail"]
    assert result["answer"] == "cannot answer"
