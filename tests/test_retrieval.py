"""Builds the real vector index in a scratch directory and checks retrieval quality
against the same known-answer questions eval/run_eval.py uses, so a regression in
extraction/chunking/embedding fails CI instead of just showing up in a manual eval run.
"""
import json
from pathlib import Path

import pytest

from eval.run_eval import grade_retrieval
from rag import ingest, store
from rag.rag import retrieve

QUESTIONS_PATH = Path(__file__).resolve().parent.parent / "eval" / "questions.json"

# Measured hit rate at time of writing is 80% (12/15); keep a margin below that so
# minor embedding-model or ranking changes don't make CI flaky.
MIN_RETRIEVAL_HIT_RATE = 0.6


@pytest.fixture(scope="session")
def indexed_corpus(tmp_path_factory):
    """Ingests the real data/raw/ corpus into a throwaway Chroma directory."""
    original_db_dir = store.DB_DIR
    store.DB_DIR = tmp_path_factory.mktemp("chroma")
    try:
        ingest.run()
        yield
    finally:
        store.DB_DIR = original_db_dir


def test_retrieval_hit_rate_meets_baseline(indexed_corpus):
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    hits = 0
    for q in questions:
        chunks = retrieve(q["question"], top_k=5)
        if grade_retrieval(chunks, q["expect_source_contains"]):
            hits += 1

    rate = hits / len(questions)
    assert rate >= MIN_RETRIEVAL_HIT_RATE, f"retrieval hit rate dropped to {rate:.0%}"


def test_retrieval_returns_relevant_chunk_for_a_known_fact(indexed_corpus):
    chunks = retrieve("What R-squared did Bunchgrass Meadow achieve?", top_k=5)
    assert any("0.954" in c["text"] for c in chunks)
