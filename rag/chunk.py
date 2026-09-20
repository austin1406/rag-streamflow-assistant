"""Turns page/slide-level extraction units into overlapping, word-bounded chunks."""

TARGET_WORDS = 180
OVERLAP_WORDS = 40


def _split_words(text):
    return text.split()


def chunk_unit(unit, target_words=TARGET_WORDS, overlap_words=OVERLAP_WORDS):
    """Splits one extraction unit (a page/slide/function) into overlapping chunks.

    Short units (the common case for slides and functions) stay as a single chunk.
    """
    words = _split_words(unit["text"])
    if len(words) <= target_words:
        return [dict(unit)]

    chunks = []
    start = 0
    part = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        chunk_text = " ".join(words[start:end])
        part += 1
        chunks.append({
            "text": chunk_text,
            "loc": f"{unit['loc']} (part {part})",
            "source": unit["source"],
        })
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


def chunk_corpus(units, target_words=TARGET_WORDS, overlap_words=OVERLAP_WORDS):
    chunks = []
    for unit in units:
        chunks.extend(chunk_unit(unit, target_words, overlap_words))

    for i, c in enumerate(chunks):
        c["id"] = f"{c['source']}::{c['loc']}".replace(" ", "_")
        if any(other["id"] == c["id"] for other in chunks[:i]):
            c["id"] = f"{c['id']}_{i}"
    return chunks
