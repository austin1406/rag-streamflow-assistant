# Streamflow RAG Assistant

A citation-grounded question-answering system over a real research artifact: my
[LSTM + Dual Adaptive Conformal Prediction](data/raw/Final_Report_LSTM_DACP.pdf) final
report on streamflow uncertainty quantification across five Washington State watersheds,
plus its two companion slide decks and the training script.

Ask a question, get an answer with inline `[n]` citations, and see the exact source
chunk (document + page/slide number) each claim was pulled from — no black box.

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
ollama pull llama3.1
ollama serve

# Option B: use an API key you already have
set ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY / GROQ_API_KEY
```

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

Full per-question results (sources retrieved + generated answer) are written to
`eval/results.json`.

## Project layout

```
rag/
  extract.py      PDF / PPTX / Python source → page/slide/function text units
  chunk.py        word-bounded chunking with overlap
  store.py        embedding model + Chroma collection
  ingest.py        end-to-end indexing script
  llm_backend.py  Ollama / Anthropic / OpenAI / Groq, chosen by env var
  rag.py          retrieval + prompt construction + citation-required generation
eval/
  questions.json  15 known-answer test questions
  run_eval.py     retrieval hit-rate + answer-accuracy grading
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
