"""Ingests everything in data/raw/ into the Chroma vector store.

Usage:
    python -m rag.ingest
"""
from pathlib import Path

from .extract import extract_corpus
from .chunk import chunk_corpus
from .store import reset_collection, embed

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def run():
    print(f"Extracting from {RAW_DIR} ...")
    units = extract_corpus(RAW_DIR)
    print(f"  {len(units)} page/slide/function units extracted")

    chunks = chunk_corpus(units)
    print(f"  {len(chunks)} chunks after splitting")

    print("Embedding chunks (all-MiniLM-L6-v2, local, no API cost) ...")
    vectors = embed([c["text"] for c in chunks])

    print("Writing to Chroma ...")
    collection = reset_collection()
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "loc": c["loc"]} for c in chunks],
    )
    print(f"Done. {collection.count()} chunks indexed in data/chroma/")


if __name__ == "__main__":
    run()
