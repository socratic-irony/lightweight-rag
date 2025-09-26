import json
from pathlib import Path

from lightweight_rag.io_biblio import load_biblio_index
from lightweight_rag.cite import pandoc_citation
from lightweight_rag.models import DocMeta


def test_load_biblio_index(tmp_path: Path):
    idx = tmp_path / ".biblio-index.json"
    data = [
        {
            "pdfFile": "Example.pdf",
            "citekey": "smith2020",
            "title": "An Example",
            "authors": [{"family": "Smith", "given": "John"}],
            "year": 2020,
            "doi": "10.1234/abc",
            "pages": {"start": 5, "end": 20}
        }
    ]
    idx.write_text(json.dumps(data), encoding="utf-8")

    m = load_biblio_index(idx)
    assert "example.pdf" in m
    e = m["example.pdf"]
    assert e.citekey == "smith2020"
    assert e.start_page == 5


def test_pandoc_citation_with_offset():
    meta = DocMeta(
        title="",
        authors=["Smith, John"],
        year=2020,
        doi=None,
        source="Example.pdf",
        start_page=10,
        end_page=12,
        citekey="smith2020",
    )
    # PDF page 3 corresponds to actual page 12
    c = pandoc_citation(meta, 3)
    assert c == "[@smith2020, p. 12]"

