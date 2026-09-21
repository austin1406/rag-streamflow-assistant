import textwrap
from pathlib import Path

from rag.extract import extract_corpus, extract_pdf, extract_pptx, extract_py

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_extract_py_splits_into_function_and_class_blocks(tmp_path):
    src = textwrap.dedent('''\
        """module docstring"""
        import os

        def foo():
            return 1

        class Bar:
            def method(self):
                pass
        ''')
    f = tmp_path / "sample.py"
    f.write_text(src, encoding="utf-8")

    units = extract_py(f)
    locs = [u["loc"] for u in units]

    assert any("foo" in loc for loc in locs)
    assert any("Bar" in loc for loc in locs)
    assert all(u["source"] == "sample.py" for u in units)


def test_extract_pdf_report_contains_known_content():
    units = extract_pdf(RAW_DIR / "Final_Report_LSTM_DACP.pdf")
    assert len(units) > 0
    assert all(u["loc"].startswith("page ") for u in units)

    all_text = " ".join(u["text"] for u in units)
    assert "Bunchgrass Meadow" in all_text
    assert "0.954" in all_text


def test_extract_pptx_returns_slide_units():
    units = extract_pptx(RAW_DIR / "LSTM_DACP_Presentation_Spring2026.pptx")
    assert len(units) > 0
    assert all(u["loc"].startswith("slide ") for u in units)


def test_extract_corpus_covers_every_supported_file():
    units = extract_corpus(RAW_DIR)
    sources = {u["source"] for u in units}
    assert "Final_Report_LSTM_DACP.pdf" in sources
    assert "lstm_train.py" in sources
    assert any(s.endswith(".pptx") for s in sources)
