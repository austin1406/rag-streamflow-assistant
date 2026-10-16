import json
import sqlite3

import pytest

from rag import eval_db
from rag.tools import MAX_SQL_ROWS, query_eval_results, search_documents


def fake_retrieve(question, top_k=5):
    return [
        {"id": f"c{i}", "text": f"text {i}", "source": "report.pdf", "loc": f"page {i}", "distance": 0.1}
        for i in range(top_k)
    ]


@pytest.fixture
def eval_db_path(tmp_path):
    questions = [
        {"question": "Q1?", "expect_source_contains": "report", "expect_answer_contains": ["0.954"]},
        {"question": "Q2?", "expect_source_contains": "missing.pdf", "expect_answer_contains": ["a", "b"]},
        {"question": "Q3?", "expect_source_contains": None, "expect_answer_contains": ["x"]},
    ]
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps(questions), encoding="utf-8")
    db_path = tmp_path / "eval.sqlite"
    eval_db.build(db_path=db_path, questions_path=questions_path, retrieve_fn=fake_retrieve)
    return db_path


def test_search_documents_returns_chunks_without_distance():
    res = search_documents({"query": "r2 for bunchgrass", "top_k": 3}, retrieve_fn=fake_retrieve)
    assert res["ok"] is True
    assert len(res["chunks"]) == 3
    assert set(res["chunks"][0]) == {"id", "source", "loc", "text"}


@pytest.mark.parametrize("args", [{}, {"query": ""}, {"query": "   "}, {"query": 5}])
def test_search_documents_rejects_bad_query(args):
    res = search_documents(args, retrieve_fn=fake_retrieve)
    assert res["ok"] is False
    assert "query" in res["error"]


@pytest.mark.parametrize("top_k", [0, 11, -1, "5", True, 2.5])
def test_search_documents_rejects_bad_top_k(top_k):
    res = search_documents({"query": "x", "top_k": top_k}, retrieve_fn=fake_retrieve)
    assert res["ok"] is False
    assert "top_k" in res["error"]


def test_eval_db_records_retrieval_results(eval_db_path):
    res = query_eval_results(
        {"sql": "SELECT question_id, retrieval_ok FROM eval_retrieval ORDER BY question_id"},
        db_path=eval_db_path,
    )
    assert res["ok"] is True
    assert res["rows"] == [[1, 1], [2, 0], [3, 1]]


def test_query_eval_results_runs_aggregates(eval_db_path):
    res = query_eval_results(
        {"sql": "SELECT COUNT(*) AS n, SUM(retrieval_ok) AS hits FROM eval_retrieval;"},
        db_path=eval_db_path,
    )
    assert res["ok"] is True
    assert res["columns"] == ["n", "hits"]
    assert res["rows"] == [[3, 2]]


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE eval_questions",
        "DELETE FROM eval_questions",
        "INSERT INTO eval_questions VALUES (99, 'x', 'y', 'z')",
        "UPDATE eval_questions SET question = 'hacked'",
        "PRAGMA table_info(eval_questions)",
        "ATTACH DATABASE 'other.db' AS other",
        "SELECT 1; DROP TABLE eval_questions",
        "SELECT * FROM eval_questions; SELECT 1",
        "",
        "   ",
    ],
)
def test_query_eval_results_rejects_anything_but_one_select(eval_db_path, sql):
    res = query_eval_results({"sql": sql}, db_path=eval_db_path)
    assert res["ok"] is False

    conn = sqlite3.connect(eval_db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM eval_questions").fetchone()[0] == 3
    finally:
        conn.close()


def test_query_eval_results_truncates_large_results(tmp_path):
    db_path = tmp_path / "big.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (n INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(MAX_SQL_ROWS + 25)])
    conn.commit()
    conn.close()

    res = query_eval_results({"sql": "SELECT n FROM t"}, db_path=db_path)
    assert res["ok"] is True
    assert len(res["rows"]) == MAX_SQL_ROWS
    assert res["truncated"] is True


def test_query_eval_results_reports_sql_errors(eval_db_path):
    res = query_eval_results({"sql": "SELECT nope FROM eval_questions"}, db_path=eval_db_path)
    assert res["ok"] is False
    assert "SQL error" in res["error"]


def test_query_eval_results_reports_missing_database(tmp_path):
    res = query_eval_results({"sql": "SELECT 1"}, db_path=tmp_path / "nope.sqlite")
    assert res["ok"] is False
    assert "not found" in res["error"]
