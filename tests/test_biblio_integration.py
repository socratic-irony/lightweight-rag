import json
from pathlib import Path

from lightweight_rag.io_biblio import load_biblio_index, load_biblio_index_by_doi, BiblioEntry
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


def test_load_biblio_index_nonexistent(tmp_path: Path):
    """Test loading biblio index from nonexistent file."""
    idx = tmp_path / "nonexistent.json"
    m = load_biblio_index(idx)
    assert m == {}


def test_load_biblio_index_invalid_json(tmp_path: Path):
    """Test loading biblio index with invalid JSON."""
    idx = tmp_path / "invalid.json"
    idx.write_text("not valid json", encoding="utf-8")
    m = load_biblio_index(idx)
    assert m == {}


def test_load_biblio_index_empty_pdf_file(tmp_path: Path):
    """Test loading biblio index with empty pdfFile."""
    idx = tmp_path / "index.json"
    data = [
        {
            "pdfFile": "",
            "citekey": "test2020",
            "title": "Test",
        }
    ]
    idx.write_text(json.dumps(data), encoding="utf-8")
    m = load_biblio_index(idx)
    assert len(m) == 0


def test_load_biblio_index_by_doi(tmp_path: Path):
    """Test loading biblio index by DOI."""
    idx = tmp_path / "index.json"
    data = [
        {
            "pdfFile": "Example.pdf",
            "citekey": "smith2020",
            "title": "An Example",
            "doi": "10.1234/ABC",
        },
        {
            "pdfFile": "Example2.pdf",
            "citekey": "jones2021",
            "title": "Another Example",
            "doi": "10.5678/DEF",
        }
    ]
    idx.write_text(json.dumps(data), encoding="utf-8")
    
    m = load_biblio_index_by_doi(idx)
    assert "10.1234/abc" in m  # Should be lowercased
    assert "10.5678/def" in m
    assert m["10.1234/abc"].citekey == "smith2020"


def test_load_biblio_index_by_doi_no_doi(tmp_path: Path):
    """Test loading biblio index by DOI when entries have no DOI."""
    idx = tmp_path / "index.json"
    data = [
        {
            "pdfFile": "Example.pdf",
            "citekey": "smith2020",
            "title": "An Example",
        }
    ]
    idx.write_text(json.dumps(data), encoding="utf-8")
    
    m = load_biblio_index_by_doi(idx)
    assert len(m) == 0


def test_biblio_entry_minimal():
    """Test BiblioEntry with minimal data."""
    row = {"pdfFile": "test.pdf"}
    entry = BiblioEntry(row)
    assert entry.pdf_file == "test.pdf"
    assert entry.citekey is None
    assert entry.title is None
    assert entry.authors == []
    assert entry.year is None
    assert entry.doi is None
    assert entry.start_page is None
    assert entry.end_page is None


def test_biblio_entry_complete():
    """Test BiblioEntry with complete data."""
    row = {
        "pdfFile": "test.pdf",
        "citekey": "test2020",
        "title": "Test Paper",
        "authors": [{"family": "Test", "given": "Author"}],
        "year": 2020,
        "doi": "10.1234/test",
        "pages": {"start": 1, "end": 10}
    }
    entry = BiblioEntry(row)
    assert entry.pdf_file == "test.pdf"
    assert entry.citekey == "test2020"
    assert entry.title == "Test Paper"
    assert len(entry.authors) == 1
    assert entry.year == 2020
    assert entry.doi == "10.1234/test"
    assert entry.start_page == 1
    assert entry.end_page == 10


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


def test_pandoc_citation_no_offset():
    """Test pandoc citation without offset."""
    meta = DocMeta(
        title="",
        authors=["Smith, John"],
        year=2020,
        doi=None,
        source="Example.pdf",
        citekey="smith2020",
    )
    c = pandoc_citation(meta, 3)
    assert c == "[@smith2020, p. 3]"


def test_pandoc_citation_no_citekey():
    """Test pandoc citation without citekey."""
    meta = DocMeta(
        title="",
        authors=["Smith, John"],
        year=2020,
        doi=None,
        source="Example.pdf",
    )
    c = pandoc_citation(meta, 3)
    assert c is None

