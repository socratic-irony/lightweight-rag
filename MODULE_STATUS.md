# Module Usage Summary

The lightweight-rag system is now fully ready for use as a Python module and subprocess integration with Node.js applications.

## ✅ What's Working

### Python Module Import
```python
import lightweight_rag

# Direct usage
config = lightweight_rag.get_default_config()
config["paths"]["pdf_dir"] = "your/pdfs"
results = lightweight_rag.query_pdfs("your query", config)

# Async usage for parallel queries
import asyncio
results = await lightweight_rag.run_rag_pipeline(config, "your query")
```

### Subprocess JSON Interface
```bash
# JSON input/output for Node.js integration
echo '{"query": "test", "config": {"paths": {"pdf_dir": "pdfs"}}}' | python -m lightweight_rag.cli_subprocess --json
```

### CLI Interface
```bash
# Direct command line usage
python -m lightweight_rag.cli_subprocess --query "test query" --pdf_dir pdfs --top_k 5
```

### Node.js Integration
See `examples/nodejs_integration.js` for a complete integration example with:
- Single query execution
- Parallel query execution
- Batch processing
- Error handling
- Timeout management

## ✅ Key Features for Node.js

1. **Clean JSON I/O**: Progress messages go to stderr, JSON response to stdout
2. **Proper Error Handling**: Standardized error responses
3. **Thread Safety**: Each subprocess runs independently
4. **Parallel Execution**: Tested with concurrent queries
5. **Configuration Flexibility**: Runtime config overrides
6. **Timeout Support**: Configurable timeouts for long-running queries

## ✅ Validated Functionality

- ✓ Module imports without permission errors
- ✓ JSON subprocess interface returns clean JSON
- ✓ CLI interface works correctly  
- ✓ Parallel execution completes successfully
- ✓ Configuration management works
- ✓ Error responses are properly formatted

## Usage from Node.js

```javascript
const rag = new LightweightRAG({
    pdfDir: './pdfs',
    cacheDir: './.rag_cache',
    topK: 8
});

// Single query
const result = await rag.query('machine learning');

// Parallel queries  
const queries = ['AI ethics', 'data privacy', 'automation'];
const results = await rag.queryParallel(queries);
```

## Files Added/Modified

- `lightweight_rag/__init__.py`: Extended exports for module usage
- `lightweight_rag/subprocess_interface.py`: JSON subprocess interface
- `lightweight_rag/cli_subprocess.py`: CLI interface with multiple modes
- `lightweight_rag/performance.py`: Fixed deterministic sorting
- `lightweight_rag/index.py`: Fixed lazy cache directory creation
- `tests/test_module_usage.py`: Comprehensive module usage tests
- `examples/nodejs_integration.js`: Complete Node.js integration example
- `MODULE_USAGE.md`: Detailed usage documentation
- `validate_module.py`: Validation script for testing

The module is now locked down and ready for production use with Node.js applications.