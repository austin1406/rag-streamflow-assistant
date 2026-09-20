"""Evaluates the RAG pipeline against a fixed set of known-answer questions.

Reports two numbers:
  - retrieval hit rate: did the expected source document show up in the top-k chunks?
  - answer accuracy: does the generated answer contain the expected fact(s)?
    (only computed if a generation backend is reachable; retrieval can be graded standalone)

Usage:
    python -m eval.run_eval [--top-k 5]
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rag.rag import retrieve, build_prompt
from rag.llm_backend import get_backend, BackendError

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"


def grade_retrieval(chunks, expect_source_contains):
    if not expect_source_contains:
        return True
    return any(expect_source_contains.lower() in c["source"].lower() for c in chunks)


def grade_answer(answer_text, expect_answer_contains):
    if answer_text is None:
        return None
    lower = answer_text.lower()
    return all(term.lower() in lower for term in expect_answer_contains)


def run(top_k=5):
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    try:
        backend = get_backend()
        backend_name = backend.name
    except BackendError as e:
        backend = None
        backend_name = None
        print(f"No generation backend reachable ({e}); grading retrieval only.\n")

    results = []
    for q in questions:
        chunks = retrieve(q["question"], top_k=top_k)
        retrieval_ok = grade_retrieval(chunks, q["expect_source_contains"])

        answer_text, answer_ok = None, None
        if backend is not None:
            prompt = build_prompt(q["question"], chunks)
            try:
                answer_text = backend.generate(prompt)
                answer_ok = grade_answer(answer_text, q["expect_answer_contains"])
            except BackendError as e:
                answer_text = f"[generation failed: {e}]"

        results.append({
            "question": q["question"],
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
            "top_sources": [f"{c['source']} {c['loc']}" for c in chunks],
            "answer": answer_text,
        })

    n = len(results)
    retrieval_hits = sum(r["retrieval_ok"] for r in results)
    print(f"Backend: {backend_name or 'none (retrieval-only run)'}")
    print(f"Retrieval hit rate: {retrieval_hits}/{n} ({100 * retrieval_hits / n:.1f}%)\n")

    if backend is not None:
        graded = [r for r in results if r["answer_ok"] is not None]
        answer_hits = sum(r["answer_ok"] for r in graded)
        if graded:
            print(f"Answer accuracy: {answer_hits}/{len(graded)} ({100 * answer_hits / len(graded):.1f}%)\n")

    for r in results:
        mark_r = "PASS" if r["retrieval_ok"] else "FAIL"
        mark_a = "PASS" if r["answer_ok"] else ("FAIL" if r["answer_ok"] is False else "-")
        print(f"[retrieval:{mark_r}] [answer:{mark_a}] {r['question']}")
        print(f"    top sources: {r['top_sources']}")
        if r["answer"]:
            print(f"    answer: {r['answer'][:200]}")
        print()

    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(top_k=args.top_k)
