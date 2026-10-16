"""Builds a small SQLite database from the evaluation questions so the agent's SQL tool has
real, checkable data to query (which questions exist, which sources they expect, and
whether retrieval found them).

Usage:
    python -m rag.eval_db
"""
import json
import sqlite3
from pathlib import Path

from .tools import EVAL_DB_PATH

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "eval" / "questions.json"

SCHEMA = """
CREATE TABLE eval_questions (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    expected_source TEXT,
    expected_terms TEXT NOT NULL
);
CREATE TABLE eval_retrieval (
    question_id INTEGER NOT NULL REFERENCES eval_questions(id),
    top_k INTEGER NOT NULL,
    retrieval_ok INTEGER NOT NULL,
    top_sources TEXT NOT NULL
);
"""


def _retrieval_ok(chunks, expected_source):
    if not expected_source:
        return True
    return any(expected_source.lower() in c["source"].lower() for c in chunks)


def build(db_path=EVAL_DB_PATH, questions_path=QUESTIONS_PATH, retrieve_fn=None, top_k=5):
    if retrieve_fn is None:
        from .rag import retrieve as retrieve_fn

    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        for i, q in enumerate(questions, start=1):
            conn.execute(
                "INSERT INTO eval_questions VALUES (?, ?, ?, ?)",
                (i, q["question"], q.get("expect_source_contains"), ", ".join(q["expect_answer_contains"])),
            )
            chunks = retrieve_fn(q["question"], top_k=top_k)
            conn.execute(
                "INSERT INTO eval_retrieval VALUES (?, ?, ?, ?)",
                (
                    i,
                    top_k,
                    int(_retrieval_ok(chunks, q.get("expect_source_contains"))),
                    "; ".join(f"{c['source']} {c['loc']}" for c in chunks),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return len(questions)


if __name__ == "__main__":
    n = build()
    print(f"Wrote {n} questions to {EVAL_DB_PATH}")
