# Installation Guide

## Installing lightweight-rag

### For End Users

#### From Source (Recommended)
```bash
git clone https://github.com/socratic-irony/lightweight-rag.git
cd lightweight-rag
pip install -e .
```

#### With Semantic Reranking
```bash
pip install -e ".[semantic]"
```

#### For Development
```bash
pip install -e ".[dev]"
```

### Quick Start After Installation

1. **Create a PDF directory**:
   ```bash
   mkdir pdfs
   # Add your PDF files to this directory
   ```

2. **Run a search**:
   ```bash
   python rag.py --query "your search query"
   ```

3. **Or use as a module**:
   ```python
   import lightweight_rag
   
   config = lightweight_rag.get_default_config()
   config["paths"]["pdf_dir"] = "./pdfs"
   results = lightweight_rag.query_pdfs("your query", config)
   ```

### Requirements

- **Python 3.10+** (required)
- **Core dependencies**: Automatically installed with package
- **Optional**: `sentence-transformers` for semantic reranking (large download)

### Verification

Test your installation:
```bash
python -c "import lightweight_rag; print('✅ Installation successful')"
lightweight-rag --help
```

### Troubleshooting

1. **Python version**: Make sure you're using Python 3.10 or higher
2. **Virtual environment**: Recommended to avoid dependency conflicts
3. **PyMuPDF issues**: Make sure you have the system dependencies for PDF processing

For more details, see the main [README.md](../README.md).