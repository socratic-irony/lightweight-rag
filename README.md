# Lightweight RAG

A minimal PDF → BM25 → top-k raw chunks with author–date–page citations.

## Features
- PDF processing and text extraction
- BM25 ranking with caching
- RM3 pseudo-relevance feedback
- Proximity and n-gram bonuses
- Diversity control
- Optional CPU semantic reranking (sentence-transformers)
- Crossref page-offset citations

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. For optional semantic reranking (larger download):
```bash
pip install numpy>=1.24.0 sentence-transformers>=2.2.0
```

## Requirements

- Python 3.10 or higher
- Core dependencies are listed in `requirements.txt`

## Usage

```bash
python lightweight-rag.py --help
```

Basic usage:
```bash
python lightweight-rag.py --pdf_dir pdfs --query "your search query"
```

See `ROADMAP.md` for planned improvements and configuration options.