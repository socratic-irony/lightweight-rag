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
- **Heuristic Reranking**: Lightweight pre-filtering based on coverage, proximity, and phrase matching

### 🧠 **Advanced Retrieval**
- **Reciprocal Rank Fusion (RRF)**: Combines multiple ranking strategies for better results
- **MMR Diversification**: Maximal Marginal Relevance for balanced result diversity
- **Diversity Control**: Prevents over-representation from single documents
- **Semantic Reranking**: Optional CPU-based sentence-transformers integration
- **Multi-Run Ranking**: Fuses baseline BM25, RM3 expansion, semantic, and robust query variants
- **Caching System**: Intelligent caching of parsed PDFs, indices, and metadata
- **Academic Citations**: Automatic DOI extraction with Crossref/OpenAlex/Unpaywall integration

### 📑 **Flexible Text Processing**
- **Sliding Window Chunking**: Configurable window size with overlap for optimal precision
- **Multiple Chunking Strategies**: Page-based, sentence-based, or sliding window modes
- **Text Quality Validation**: Automatic detection and handling of unreadable PDFs

### ⚡ **Performance & Usability**
- **Fast Cold Start**: Only imports what's needed for quick startup
- **Memory Efficient**: Streaming PDF processing with configurable chunk sizes
- **Configurable**: YAML config with environment variable and CLI overrides
- **Robust**: Comprehensive error handling and graceful degradation
- **Module & Subprocess Interfaces**: Use as Python package or via JSON subprocess (Node.js compatible)

## 🎯 Quick Start

### Basic Usage

**When installed as a package** (the `lightweight-rag` console script wraps the JSON/subprocess interface and also supports direct queries):

```bash
# Search your PDF collection
lightweight-rag --query "machine learning algorithms"

# Use a specific directory
lightweight-rag --pdf_dir ./research_papers --query "deep learning" --pretty

# JSON subprocess mode (for Node.js integration): reads a JSON request from stdin
echo '{"query": "neural networks", "config": {"paths": {"pdf_dir": "./pdfs"}}}' | lightweight-rag
```

**Running directly from source** (`rag.py` is the full human-facing CLI with all tuning flags):

```bash
# Search your PDF collection
python rag.py --query "machine learning algorithms"

# Use a specific directory
python rag.py --pdf_dir ./research_papers --query "deep learning"

# Enable query expansion for better recall
python rag.py --rm3 --query "neural networks"
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
    enabled: false     # Off by default; enable with sentence-transformers installed
    model: "sentence-transformers/all-MiniLM-L6-v2"
  final_top_k: 8      # Results to return

fusion:
  rrf:
    enabled: true      # Enable Reciprocal Rank Fusion
    C: 60              # RRF parameter (higher = less aggressive fusion)
    cap: 200           # Max items per run to consider in fusion
  robust_query:
    enabled: true      # Enable robust query variant run

performance:
  api_semaphore_size: 5      # Max concurrent API calls
  pdf_thread_workers: null   # None = auto (num_cores), or set manually  
  deterministic: true        # Enable deterministic tie-breaking
  numpy_seed: 42            # Seed for numpy random operations
```

### Environment Variables
A fixed set of settings can be overridden with environment variables (not all settings are supported):
```bash
export RAG_PATHS_PDF_DIR="./my_papers"
export RAG_PATHS_CACHE_DIR=".rag_cache"
export RAG_BM25_K1="2.0"
export RAG_BM25_B="0.75"
export RAG_PRF_ENABLED="true"
export RAG_PRF_FB_DOCS="6"
export RAG_PRF_FB_TERMS="10"
export RAG_RERANK_FINAL_TOP_K="8"
```

> **Note:** Only the variables listed above are supported. For other settings, edit `config.yaml` directly.

### Command Line Options
```bash
python rag.py \
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

### 🔄 Reciprocal Rank Fusion (RRF)

RRF combines multiple ranking strategies to improve result quality:

1. **Baseline BM25**: Standard BM25 scoring with proximity and n-gram bonuses
2. **RM3 Expansion**: Query expansion using pseudo-relevance feedback (if enabled)
3. **Semantic Reranking**: Sentence-transformer based reranking (if enabled)
4. **Robust Query**: Normalized query variant to catch phrasing variations

Each strategy produces a ranked list of candidates. RRF then combines these using the formula:

```
RRF_score(d) = Σ(1 / (C + rank_i(d)))
```

Where:
- `d` is a document
- `rank_i(d)` is the rank of document `d` in run `i`
- `C` is the fusion parameter (default: 60)

This approach is particularly effective when different ranking methods find complementary relevant documents.

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

**Current Coverage:** 66% (204 tests passing, 3 skipped)

## 📊 Code Quality

This repository maintains high code quality standards:

- ✅ **Average Complexity:** 6.78 (Grade B - Good)
- ✅ **Security:** 0 vulnerabilities (Bandit scan)
- ✅ **Maintainability:** 16/18 modules rated 'A'
- ✅ **Test-to-Code Ratio:** 1.11:1 (Excellent)
- ✅ **Code Style:** Black + isort formatting

**Quality Reports:**
- 📖 [Comprehensive Analysis Report](docs/CODE_QUALITY_REPORT.md)
- ✅ [Improvement Checklist](docs/CODE_QUALITY_CHECKLIST.md)
- 📊 [Quality Documentation](docs/README_CODE_QUALITY.md)

Run quality checks:
```bash
# Quick quality check
python scripts/check_quality.py

# Code formatting
black lightweight_rag/ --line-length 100
isort lightweight_rag/ --profile black

# Linting
flake8 lightweight_rag/ --max-line-length=100

# Complexity analysis
radon cc lightweight_rag/ -a -s
```

## 🔧 Advanced Usage

### Python Module Import
```python
import lightweight_rag

# Simple usage with default config
config = lightweight_rag.get_default_config()
config["paths"]["pdf_dir"] = "./my_papers"
results = lightweight_rag.query_pdfs("machine learning", config)

# Async usage for better performance
import asyncio
results = await lightweight_rag.run_rag_pipeline(config, "deep learning")
```

### Installation as Package
```bash
# Install from source
pip install -e .

# Install with semantic reranking support
pip install -e ".[semantic]"

# Install development dependencies
pip install -e ".[dev]"
```

### Using as Subprocess from Node.js
```javascript
const { spawn } = require('child_process');

function queryPDF(query, config = {}) {
  return new Promise((resolve, reject) => {
    const process = spawn('python', ['-m', 'lightweight_rag.cli_subprocess', '--json']);
    
    let output = '';
    process.stdout.on('data', (data) => output += data);
    process.on('close', (code) => {
      if (code === 0) resolve(JSON.parse(output));
      else reject(new Error(`Process failed: ${code}`));
    });
    
    process.stdin.write(JSON.stringify({ query, config }));
    process.stdin.end();
  });
}
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
