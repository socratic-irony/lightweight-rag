"""Utilities for loading and applying a bibliography index built from BibTeX.

The index is a normalized JSON file containing one row per attached PDF:

[
  {
    "pdfFile": "Some Paper.pdf",
    "citekey": "smith2020paper",
    "title": "Some Paper",
    "authors": [{"family": "Smith", "given": "John"}],
    "year": 2020,
    "doi": "10.1234/abc",
    "pages": {"start": 12, "end": 28}
  }
]

We map entries by lowercase basename for robust matching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional


class BiblioEntry:
    def __init__(self, row: Dict[str, Any]):
        self.pdf_file: str = row.get("pdfFile") or ""
        self.citekey: Optional[str] = row.get("citekey")
        self.title: Optional[str] = row.get("title")
        self.authors = row.get("authors") or []  # [{family, given}]
        self.year: Optional[int] = row.get("year")
        self.doi: Optional[str] = row.get("doi")
        pages = row.get("pages") or {}
        self.start_page: Optional[int] = pages.get("start") if isinstance(pages, dict) else None
        self.end_page: Optional[int] = pages.get("end") if isinstance(pages, dict) else None


def load_biblio_index(path: str | Path) -> Dict[str, BiblioEntry]:
    """Load the bibliography index JSON and return a map by lowercased basename."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: Dict[str, BiblioEntry] = {}
    if isinstance(data, list):
        for row in data:
            try:
                entry = BiblioEntry(row)
                if not entry.pdf_file:
                    continue
                key = Path(entry.pdf_file).name.lower()
                out[key] = entry
            except Exception:
                continue
    return out

