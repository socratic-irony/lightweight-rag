# 🚀 Lightweight RAG

[![Tests](https://github.com/socratic-irony/lightweight-rag/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/socratic-irony/lightweight-rag/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/socratic-irony/lightweight-rag/branch/main/graph/badge.svg)](https://codecov.io/gh/socratic-irony/lightweight-rag)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A fast, minimal PDF → BM25 → top-k retrieval system with smart caching, query expansion, and academic citations.

## ✨ Features

### 🔍 **Smart Search & Ranking**
- **BM25 Algorithm**: Industry-standard lexical search with configurable parameters
- **RM3 Query Expansion**: Pseudo-relevance feedback for improved recall
- **Proximity Bonuses**: Rewards when query terms appear close together
- **N-gram Matching**: Boosts results containing exact phrase matches
- **Pattern Recognition**: Identifies academic answer patterns ("we propose", "method")

### 🧠 **Advanced Retrieval**
- **Diversity Control**: Prevents over-representation from single documents
- **Semantic Reranking**: Optional CPU-based sentence-transformers integration
- **Caching System**: Intelligent caching of parsed PDFs, indices, and metadata
- **Academic Citations**: Automatic DOI extraction with Crossref/OpenAlex integration

### ⚡ **Performance & Usability**
- **Fast Cold Start**: Only imports what's needed for quick startup
- **Memory Efficient**: Streaming PDF processing with configurable chunk sizes
- **Configurable**: YAML config with environment variable and CLI overrides
- **Robust**: Comprehensive error handling and graceful degradation

## 🎯 Quick Start

### Basic Usage

```bash
# Search your PDF collection
python lightweight-rag.py --query "machine learning algorithms"

# Use a specific directory
python lightweight-rag.py --pdf_dir ./research_papers --query "deep learning"

# Enable query expansion for better recall
python lightweight-rag.py --rm3 --query "neural networks"
```

### Example Output

```json
[
  {
    "text": "We propose a novel deep learning architecture that achieves state-of-the-art performance...",
    "citation": "Smith et al. 2023, p. 42",
    "source": {
      "file": "/papers/smith2023_deep_learning.pdf",
      "page": 42,
      "doi": "10.1038/s41586-023-05881-4",
      "title": "Deep Learning Architectures for Computer Vision"
    },
    "score": 2.847
  }
]
```

## 🛠️ Installation

### Core Dependencies
```bash
pip install -r requirements.txt
```

### Optional: Semantic Reranking
For enhanced semantic search capabilities:
```bash
pip install numpy>=1.24.0 sentence-transformers>=2.2.0
```

*Note: Semantic reranking significantly increases download size (~500MB) but provides better result ranking.*

## ⚙️ Configuration

### Quick Configuration
Create a `config.yaml` file to customize behavior:

```yaml
paths:
  pdf_dir: "research_papers"
  cache_dir: ".rag_cache"

bm25:
  k1: 1.5              # Term frequency saturation
  b: 0.75              # Document length normalization

prf:
  enabled: true        # Enable RM3 query expansion
  fb_docs: 6          # Feedback documents
  fb_terms: 10        # Expansion terms

bonuses:
  proximity:
    enabled: true
    window: 30         # Token window for proximity
    weight: 0.2        # Proximity bonus weight
  
  ngram:
    enabled: true
    weight: 0.1        # N-gram match bonus

rerank:
  semantic:
    enabled: true      # Requires sentence-transformers
    model: "sentence-transformers/all-MiniLM-L6-v2"
  final_top_k: 8      # Results to return
```

### Environment Variables
Override any setting with environment variables:
```bash
export RAG_PATHS_PDF_DIR="./my_papers"
export RAG_BM25_K1="2.0"
export RAG_PRF_ENABLED="true"
```

### Command Line Options
```bash
python lightweight-rag.py \
  --pdf_dir ./papers \
  --k 10 \
  --rm3 \
  --semantic_rerank \
  --prox_lambda 0.3 \
  --ngram_lambda 0.15
```

## 🏗️ Key Components

- **Ingestion**: PyMuPDF for fast, accurate PDF text extraction
- **Indexing**: BM25 with configurable parameters and intelligent caching
- **Search**: Multi-stage ranking with lexical → bonus → semantic → diversity
- **Enrichment**: DOI detection, Crossref lookup, and citation formatting

## 🧪 Testing

This project includes a comprehensive test suite covering:

- ✅ **Core Functions**: Tokenization, scoring, pattern matching
- ✅ **Configuration System**: YAML loading, environment overrides, CLI precedence  
- ✅ **Caching Logic**: Manifest validation, corpus serialization, BM25 persistence
- ✅ **Integration Tests**: End-to-end pipeline validation
- ✅ **Error Handling**: Graceful degradation and edge cases

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing

# Run specific test categories
python -m pytest tests/test_core_functions.py -v
python -m pytest tests/test_integration.py -v
```

## 🔧 Advanced Usage

### Batch Processing
```python
# Python API usage (if extended)
from lightweight_rag import query_pdfs

config = load_full_config("config.yaml")
results = query_pdfs("your query", config)
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with:
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) for BM25 implementation
- [sentence-transformers](https://www.sbert.net/) for semantic search
- [Crossref API](https://www.crossref.org/services/metadata-delivery/) for citation metadata

---

<div align="center">
  <strong>⭐ Star this repo if you find it useful! ⭐</strong>
</div>
