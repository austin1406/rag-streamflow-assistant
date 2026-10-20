# Streamflow RAG Assistant

[![Tests](https://github.com/austin1406/rag-streamflow-assistant/actions/workflows/test.yml/badge.svg)](https://github.com/austin1406/rag-streamflow-assistant/actions/workflows/test.yml)
[![Deploy](https://github.com/austin1406/rag-streamflow-assistant/actions/workflows/deploy.yml/badge.svg)](https://github.com/austin1406/rag-streamflow-assistant/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A citation-grounded question-answering system over a real research artifact: my
[LSTM + Dual Adaptive Conformal Prediction](data/raw/Final_Report_LSTM_DACP.pdf) final
report on streamflow uncertainty quantification across five Washington State watersheds,
plus its two companion slide decks and the training script.

Ask a question, get an answer with inline `[n]` citations, and see the exact source
chunk (document + page/slide number) each claim was pulled from — no black box.

## Live demo

Deployed on an AWS EC2 instance (Ubuntu, Docker) by the GitHub Actions pipeline described in
[DEPLOY.md](DEPLOY.md): **http://18.116.112.248/**

- It runs on a small server with a free-tier LLM key, so it can be slow, rate-limited, or offline.
  It is HTTP only, and the address changes if the server is restarted.
- Endpoints: `GET /health`, `POST /api/query` (one-shot answer with citations), and
  `POST /api/agent` (tool-calling agent; see below).
- Every push to `master` that passes the tests rebuilds the Docker image and redeploys it.

## Why this exists

This report went through several iterations (early feature-set experiments in one deck,
revised numbers in another, final numbers in the report itself) — the same fact, like
Swift Creek's R², literally has different values in different documents depending on
which feature set was used at the time. A flat-file search can't tell those apart. A RAG
pipeline that cites *which document and page* an answer came from can.

## Architecture

```
data/raw/*.pdf,*.pptx,*.py
        │  extract.py   (pymupdf / python-pptx / regex block splitter)
        ▼
   page/slide/function-level text units
        │  chunk.py     (~180-word chunks, 40-word overlap)
        ▼
   chunk.py output ──▶ store.py (sentence-transformers all-MiniLM-L6-v2, local, no API cost)
        ▼
   Chroma persistent vector store (data/chroma/)
        ▲
        │  rag.py: embed question → top-k retrieval → prompt w/ numbered context
        │
   llm_backend.py (Ollama / Anthropic / OpenAI / Groq — pluggable via env var)
        ▼
   app.py (Flask) ──▶ templates/index.html + static/app.js
```

Every layer is swappable independently: change the embedding model in `store.py`,
change chunk size in `chunk.py`, or change the generation backend without touching
retrieval at all.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
```

Copy `.env.example` to `.env` (or just set env vars directly) and pick a generation
backend — **Ollama is free and fully local**:

```bash
# Option A: free, local, zero API cost
# install from https://ollama.com, then:
ollama pull llama3.2:1b   # or llama3.1 (4.9GB) if you have >6GB free RAM
ollama serve

# Option B: use an API key you already have
set ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY / GROQ_API_KEY
```

`.env` is auto-loaded (`python-dotenv`) so you don't need to export env vars manually.

> **Model size note:** `OLLAMA_MODEL` defaults to `llama3.1` (8B, needs ~5GB free RAM to
> load). On a machine with less headroom, `ollama pull llama3.2:1b` and set
> `OLLAMA_MODEL=llama3.2:1b` in `.env` — that's what this repo's own `.env` uses. Smaller
> models follow the citation format less reliably (see eval results below); an API
> backend (Anthropic/OpenAI/Groq) or a bigger local model will score higher.

Build the vector index (run once, or again after adding documents to `data/raw/`):

```bash
python -m rag.ingest
```

Run the app:

```bash
python app.py
```

Open http://localhost:5000. Retrieval works immediately with no backend configured —
you'll see the retrieved source chunks with distances; the generated answer requires
one of the backends above.

## Tool-calling agent

`POST /api/agent` runs a small agent loop (`rag/agent.py`) instead of one-shot retrieval. The
model can call two tools, then return structured output:

- `search_documents(query, top_k)`: retrieval over the vector store; results are numbered
  passages the final answer cites.
- `query_eval_results(sql)`: one read-only `SELECT` over a SQLite database of the evaluation
  questions and retrieval results (`python -m rag.eval_db` builds it).

```bash
curl -X POST localhost:5000/api/agent -H "Content-Type: application/json" \
  -d '{"question": "How many evaluation questions did retrieval get right?"}'
```

Returns `answer`, `citations` (number, source, location), `chunks`, `tool_calls`, `steps`,
and `warnings`. The protocol is plain JSON replies, so it needs only each backend's existing
`generate(prompt)` and does not depend on any one vendor's tool-calling API.

**Measured on the 15 evaluation questions (answers graded by checking for the expected fact):**

| Model | Plain one-shot pipeline | Agent loop |
|---|---|---|
| `llama3.2:1b` (free, local via Ollama) | 6/15 (40%) | 1/15 |
| `openai/gpt-oss-120b` (Groq) | 10/15 (67%) | 2/15 |
| `qwen/qwen3.8-27b` (Groq, the default) | **12/15 (80%)** | **11/15 (73%)** |

With Qwen the model called a tool by itself on 15/15 questions and answered 13/15 (the other 2
returned no final answer). The agent's failures with the other two models were instructive:
the 1B model is too small to reason through an agent loop, and `gpt-oss` is built around its own
tool-calling format, so it wrote its output into a hidden reasoning field or skipped the search
and answered from memory. Use a model that follows plain JSON instructions for agent mode, or
`/api/query` with small local models. Getting there required a worked example in the prompt,
tolerant citation parsing (small models quote the numbers), and an automatic search when a
model answers without calling a tool. The Groq backend also waits and retries on rate limits.
Numbers are single runs on a small question set, so treat them as indicative, not a benchmark.

Guardrails, each covered by tests: tools are whitelisted and their arguments validated; the
SQL tool allows a single `SELECT` only, on a read-only connection with a SQLite authorizer,
row limit, and time limit; the loop stops after 4 steps; and any citation that does not match
a passage the agent actually retrieved is dropped and reported in `warnings`.

## Evaluation history API (Spring Boot)

`eval-api/` is a small Java service that stores every evaluation run so hit rates can be
compared over time (different models, different `top_k`). It is a Spring Boot REST API
(Spring Data JPA, Bean Validation, H2) with its own test suite; see
[eval-api/README.md](eval-api/README.md).

```bash
cd eval-api && ./mvnw spring-boot:run          # http://localhost:8080
python -m eval.run_eval                         # writes eval/results.json
python -m eval.publish_results --label baseline --model llama3.2:1b
# Stored run 1 'baseline': retrieval 12/15 (80%), answers 6/15 (40%)
```

`eval/publish_results.py` posts the results of `run_eval` to the API, and reports a clear error
if the API is down or rejects the run.

## Deployment

The app ships as a Docker image (index and embedding model baked in, served by gunicorn), and
`.github/workflows/deploy.yml` builds it, publishes it to GitHub Container Registry, and
deploys it to an AWS EC2 Ubuntu server over SSH after the tests pass. See
[DEPLOY.md](DEPLOY.md) for setup.

## Evaluation

`eval/questions.json` holds 15 known-answer questions grounded in specific numbers from
the final report (R² per site, DACP coverage rates, k_adap value, LSTM architecture
details, etc.) — the same evidence discipline used elsewhere in this project: report a
real, checkable number, not a vibe.

```bash
python -m eval.run_eval
```

`run_eval.py` grades two independent things:

- **Retrieval hit rate** — did the correct source document appear in the top-5 chunks?
  This needs no LLM and runs offline. Current result: **12/15 (80%)**. The 3 misses are
  cases where the presentations restate the same fact and rank above the final report
  (e.g. "who introduced DACP" is answered correctly, just from a slide instead of the
  report page) — a real, honest limitation, not a scoring artifact.
- **Answer accuracy** — does the generated answer actually contain the expected fact?
  Only computed when a generation backend is reachable, since it needs real generation.
  Measured end-to-end with `llama3.2:1b` (the free local model that fits this machine's
  available RAM): **6/15 (40%)**. The failures are mostly citation-formatting slips
  (`"[1] [1]"` with no actual text, or an unfilled `[n]` placeholder) rather than wrong
  facts — a known weakness of 1B-parameter models at instruction-following, not a
  retrieval problem (retrieval for those same questions still hits 80%+). Swapping in
  `llama3.1` (8B) or an API backend (Anthropic/OpenAI/Groq) measurably improves this;
  it's a hardware tradeoff, not an architecture limitation, and the number is reported
  as measured rather than rounded up.

Full per-question results (sources retrieved + generated answer) are written to
`eval/results.json`.

## Testing & CI

```bash
pip install -r requirements.txt   # includes pytest
pytest -v
```

89 Python tests (plus 26 Java tests in `eval-api/`) across these layers:
- **`tests/test_chunk.py`, `tests/test_extract.py`** — unit tests for the pure chunking
  logic and the PDF/PPTX/Python extraction functions, run against the real files in
  `data/raw/` (not mocks), asserting known content actually gets extracted correctly.
- **`tests/test_llm_backend.py`** — provider auto-detection and error-handling logic for
  the pluggable backend, with no network calls.
- **`tests/test_retrieval.py`** — an integration test that builds the real vector index
  into a scratch directory and asserts retrieval hit-rate stays at or above 60% (measured
  baseline is 80%; the margin absorbs minor embedding-model drift without going flaky).
  This formalizes what `eval/run_eval.py` measures manually into an assertion that fails
  the build on a real regression.
- **`tests/test_tools.py`, `tests/test_agent.py`** — the agent's tools and loop, driven by a
  scripted fake backend (no network, no API keys): reply parsing, citation mapping,
  hallucinated-citation handling, retries, the step limit, and SQL safety (every write or
  multi-statement query is rejected and the database is verified unchanged).
- **`tests/test_app.py`** — `/health` and `/api/agent` endpoint behavior.
- **`tests/test_publish_results.py`** — payload mapping and error handling for publishing
  evaluation runs to the Spring Boot API.

`.github/workflows/test.yml` runs the full suite on every push and pull request via
GitHub Actions (CPU-only torch, cached pip + HuggingFace model downloads).

## Project layout

```
rag/
  extract.py      PDF / PPTX / Python source → page/slide/function text units
  chunk.py        word-bounded chunking with overlap
  store.py        embedding model + Chroma collection
  ingest.py        end-to-end indexing script
  llm_backend.py  Ollama / Anthropic / OpenAI / Groq, chosen by env var
  rag.py          retrieval + prompt construction + citation-required generation
  agent.py        tool-calling loop with structured output and citation guardrails
  tools.py        search_documents and a read-only SQL tool
  eval_db.py      builds the SQLite database the SQL tool queries
eval/
  questions.json  15 known-answer test questions
  run_eval.py     retrieval hit-rate + answer-accuracy grading
  publish_results.py  sends a run's results to the eval-api service
eval-api/         Spring Boot service that stores evaluation runs (own tests, own CI job)
tests/            pytest suite (unit + retrieval integration tests)
.github/workflows/test.yml   CI: runs pytest on every push/PR
.github/workflows/eval-api.yml  CI: runs the eval-api Maven tests when eval-api/ changes
.github/workflows/deploy.yml CD: builds the image, pushes to GHCR, deploys to EC2
Dockerfile, deploy/          container image and server scripts (see DEPLOY.md)
app.py            Flask API + UI
templates/, static/
data/raw/         source documents (report PDF, 2 slide decks, training script)
```

## Notes on the underlying research

Two people (myself and Pranav Cheedalla) coupled an LSTM (2 layers × 128 hidden units,
14-day input window) with Dual Adaptive Conformal Prediction (Zong et al. 2026) to
produce calibrated prediction intervals for daily streamflow across five snowmelt-driven
watersheds. R² ranged from 0.405 (Swift Creek, glacially-driven and inherently noisy) to
0.954 (Bunchgrass Meadow). DACP coverage tracked its nominal targets closely on 3 of 5
sites and under-covered on the other 2 — the report traces this to a single conservative
`k_adap=0.005` held constant across targets instead of the paper's per-target schedule.
This assistant makes that whole analysis queryable instead of requiring a PDF read-through.
