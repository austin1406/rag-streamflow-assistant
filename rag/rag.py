"""Core retrieve-then-generate logic, shared by the Flask app and the eval script."""
from .store import get_collection, embed
from .llm_backend import get_backend, BackendError

SYSTEM_PROMPT = """You are a research assistant answering questions about a hydrology \
report on LSTM streamflow prediction with Dual Adaptive Conformal Prediction (DACP) \
uncertainty quantification, plus its companion slide decks and training script.

Rules:
- Answer using ONLY the provided context chunks. Do not use outside knowledge.
- After every factual claim, cite the chunk it came from like [1], [2], etc., matching \
the numbered context blocks below.
- If the context does not contain the answer, say so explicitly instead of guessing."""


def retrieve(question: str, top_k: int = 5):
    collection = get_collection()
    query_vec = embed([question])[0]
    result = collection.query(query_embeddings=[query_vec], n_results=top_k)

    chunks = []
    for i in range(len(result["ids"][0])):
        chunks.append({
            "id": result["ids"][0][i],
            "text": result["documents"][0][i],
            "source": result["metadatas"][0][i]["source"],
            "loc": result["metadatas"][0][i]["loc"],
            "distance": result["distances"][0][i],
        })
    return chunks


def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n".join(
        f"[{i + 1}] (source: {c['source']}, {c['loc']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return (
        f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\n"
        f"Question: {question}\n\nAnswer, with inline [n] citations:"
    )


def answer(question: str, top_k: int = 5):
    chunks = retrieve(question, top_k=top_k)
    prompt = build_prompt(question, chunks)
    try:
        backend = get_backend()
        text = backend.generate(prompt)
        error = None
    except BackendError as e:
        text = None
        error = str(e)
    return {"question": question, "chunks": chunks, "answer": text, "error": error}
