from rag.chunk import chunk_unit, chunk_corpus


def test_short_unit_stays_a_single_chunk():
    unit = {"text": "short piece of text", "loc": "slide 1", "source": "deck.pptx"}
    chunks = chunk_unit(unit, target_words=180, overlap_words=40)
    assert len(chunks) == 1
    assert chunks[0]["text"] == unit["text"]


def test_long_unit_splits_with_consistent_overlap():
    words = [f"word{i}" for i in range(500)]
    unit = {"text": " ".join(words), "loc": "page 1", "source": "report.pdf"}
    chunks = chunk_unit(unit, target_words=180, overlap_words=40)

    assert len(chunks) == 4
    for i in range(len(chunks) - 1):
        tail = chunks[i]["text"].split()[-40:]
        head = chunks[i + 1]["text"].split()[:40]
        assert tail == head, f"overlap broken between chunk {i} and {i + 1}"

    # every word in the source should show up in at least one chunk
    covered = set()
    for c in chunks:
        covered.update(c["text"].split())
    assert covered == set(words)


def test_chunk_corpus_assigns_unique_ids_even_with_duplicate_loc():
    units = [
        {"text": "hello world", "loc": "page 1", "source": "x.pdf"},
        {"text": "goodbye world", "loc": "page 1", "source": "x.pdf"},
    ]
    chunks = chunk_corpus(units)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
