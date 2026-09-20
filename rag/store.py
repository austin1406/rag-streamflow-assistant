"""Wraps the Chroma persistent collection and the local embedding model."""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
COLLECTION_NAME = "streamflow_report"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def reset_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.get_or_create_collection(COLLECTION_NAME)


def embed(texts):
    return get_embedder().encode(list(texts), show_progress_bar=False, normalize_embeddings=True).tolist()
