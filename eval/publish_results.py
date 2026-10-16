"""Publishes the latest evaluation results (eval/results.json) to the Spring Boot evaluation
API in eval-api/, so every run is stored and its hit rates can be compared over time.

Usage:
    python -m eval.run_eval
    python -m eval.publish_results --label baseline --model llama3.2:1b [--top-k 5] [--url http://localhost:8080]
"""
import argparse
import json
import sys
from pathlib import Path

import requests

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"
DEFAULT_URL = "http://localhost:8080"


def build_payload(results, label, model, top_k):
    return {
        "label": label,
        "model": model,
        "topK": top_k,
        "results": [
            {
                "question": r["question"],
                "retrievalOk": bool(r["retrieval_ok"]),
                "answerOk": r["answer_ok"],
            }
            for r in results
        ],
    }


def publish(payload, url=DEFAULT_URL):
    response = requests.post(f"{url.rstrip('/')}/api/runs", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--label", required=True, help="name for this run, e.g. baseline")
    parser.add_argument("--model", required=True, help="generation model used, e.g. llama3.2:1b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    if not args.results.exists():
        print(f"{args.results} not found. Run `python -m eval.run_eval` first.", file=sys.stderr)
        return 1

    payload = build_payload(json.loads(args.results.read_text(encoding="utf-8")), args.label, args.model, args.top_k)
    try:
        created = publish(payload, args.url)
    except requests.exceptions.ConnectionError:
        print(f"Could not reach the evaluation API at {args.url}. Start it with `cd eval-api && ./mvnw spring-boot:run`.", file=sys.stderr)
        return 1
    except requests.exceptions.HTTPError as e:
        print(f"The API rejected the run: {e.response.status_code} {e.response.text}", file=sys.stderr)
        return 1

    s = created["summary"]
    answers = "not graded" if s["answerRate"] is None else f"{s['answerHits']}/{s['answersGraded']} ({s['answerRate']:.0%})"
    print(f"Stored run {s['id']} '{s['label']}': retrieval {s['retrievalHits']}/{s['questions']} ({s['retrievalRate']:.0%}), answers {answers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
