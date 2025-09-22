"""PDF text extraction and document processing."""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from tqdm import tqdm

from .models import DocMeta, Chunk, find_doi_in_text
# Import moved inside function to avoid circular dependency


# -------------------------
# PDF Text Extraction
# -------------------------

def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from each page of a PDF."""
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            pages.append({
                "page_number": page_num + 1,  # 1-based
                "text": text
            })
        doc.close()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return pages


async def build_corpus(pdf_dir: Path) -> List[Chunk]:
    """Build corpus by extracting text from all PDFs in directory."""
    import httpx
    
    pdf_files = glob.glob(str(pdf_dir / "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        return []

    corpus = []
    doc_id = 0

    async with httpx.AsyncClient() as client:
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pages = extract_pdf_pages(pdf_file)
            if not pages:
                continue

            # Look for DOI in first 2 pages
            first_pages_text = " ".join([p["text"] for p in pages[:2]])
            doi = find_doi_in_text(first_pages_text)

            # Fetch metadata if DOI found
            meta = DocMeta(
                title=None,
                authors=[],
                year=None,
                doi=doi,
                source=os.path.basename(pdf_file)
            )

            if doi:
                # Import here to avoid circular dependency
                from .cite import crossref_meta_for_doi
                crossref_meta = await crossref_meta_for_doi(client, doi)
                if crossref_meta:
                    meta.title = crossref_meta.title
                    meta.authors = crossref_meta.authors
                    meta.year = crossref_meta.year
                    meta.start_page = crossref_meta.start_page

            # Create chunks for each page
            for page_data in pages:
                text = page_data["text"].strip()
                if text:  # Skip empty pages
                    chunk = Chunk(
                        doc_id=doc_id,
                        source=os.path.basename(pdf_file),
                        page=page_data["page_number"],
                        text=text,
                        meta=meta
                    )
                    corpus.append(chunk)

            doc_id += 1

    print(f"Extracted {len(corpus)} chunks from {len(pdf_files)} PDFs")
    return corpus