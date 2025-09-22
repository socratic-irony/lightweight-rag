# Using Lightweight RAG as a Module

The Lightweight RAG system is designed to be easily imported and used as a Python module, and can also be used as a subprocess from other programming languages like Node.js.

## Python Module Usage

### Basic Import

```python
import lightweight_rag

# Get default configuration
config = lightweight_rag.get_default_config()
config["paths"]["pdf_dir"] = "your/pdf/directory"
config["paths"]["cache_dir"] = "./cache"

# Execute a search
results = lightweight_rag.query_pdfs("your search query", config)
```

### Async Usage

```python
import asyncio
import lightweight_rag

async def async_search():
    config = lightweight_rag.get_default_config()
    config["paths"]["pdf_dir"] = "your/pdf/directory"
    
    # Use the async pipeline directly for more control
    results = await lightweight_rag.run_rag_pipeline(config, "your query")
    return results

# Run the async function
results = asyncio.run(async_search())
```

### Configuration Management

```python
import lightweight_rag

# Load from file with overrides
base_config = lightweight_rag.load_config("config.yaml")
custom_config = {
    "rerank": {"final_top_k": 10},
    "paths": {"pdf_dir": "custom/path"}
}
merged_config = lightweight_rag.merge_configs(base_config, custom_config)

results = lightweight_rag.query_pdfs("query", merged_config)
```

### Parallel Processing in Python

```python
import asyncio
import lightweight_rag

async def parallel_queries():
    config = lightweight_rag.get_default_config()
    config["paths"]["pdf_dir"] = "pdfs"
    
    queries = ["query1", "query2", "query3"]
    tasks = [lightweight_rag.run_rag_pipeline(config, query) for query in queries]
    
    results = await asyncio.gather(*tasks)
    return results

# Execute parallel queries
all_results = asyncio.run(parallel_queries())
```

## Subprocess Usage

### JSON Interface

The module provides a JSON-based subprocess interface that's perfect for integration with other languages:

```bash
# JSON input via stdin
echo '{"query": "machine learning", "config": {"paths": {"pdf_dir": "./pdfs"}}}' | python -m lightweight_rag.cli_subprocess --json
```

### Direct Command Line

```bash
# Direct query mode
python -m lightweight_rag.cli_subprocess --query "machine learning" --pdf_dir ./pdfs --pretty
```

### Batch Processing

```bash
# Batch processing from file
python -m lightweight_rag.cli_subprocess --batch queries.json --output results.json --pretty
```

## Node.js Integration

### Installation and Setup

1. Ensure Python and the lightweight_rag package are installed
2. Copy the `examples/nodejs_integration.js` file to your Node.js project
3. Use the LightweightRAG class:

```javascript
const LightweightRAG = require('./nodejs_integration.js');

const rag = new LightweightRAG({
    pdfDir: './pdfs',
    cacheDir: './.rag_cache',
    topK: 8,
    timeout: 120000
});

// Single query
const result = await rag.query('machine learning');
console.log(`Found ${result.count} results`);

// Parallel queries
const queries = ['AI ethics', 'data privacy', 'automation'];
const results = await rag.queryParallel(queries);
```

### Error Handling

```javascript
try {
    const result = await rag.query('complex query');
    console.log('Success:', result.results);
} catch (error) {
    console.error('Search failed:', error.message);
}
```

### Configuration Options

```javascript
const rag = new LightweightRAG({
    pythonPath: 'python3',  // Python executable path
    pdfDir: './documents',  // PDF directory
    cacheDir: './.cache',   // Cache directory
    topK: 10,              // Number of results
    timeout: 300000,       // Timeout in milliseconds
    config: {              // Additional RAG config
        rerank: {
            semantic: { enabled: true }
        },
        performance: {
            deterministic: true
        }
    }
});
```

## Response Format

All interfaces return results in this standardized format:

```json
{
  "success": true,
  "query": "your search query",
  "results": [
    {
      "text": "Relevant text snippet...",
      "citation": "(Author, Year, p. Page)",
      "source": {
        "file": "document.pdf",
        "page": 5,
        "doi": "10.1000/example",
        "title": "Document Title"
      },
      "score": 4.52
    }
  ],
  "count": 8,
  "error": null
}
```

### Error Response Format

```json
{
  "success": false,
  "query": "your search query",
  "results": [],
  "count": 0,
  "error": "Error description"
}
```

## Thread Safety and Parallel Execution

The module is designed to handle parallel execution safely:

### Python Async Safety
- Uses asyncio for proper concurrent execution
- Each query gets its own configuration instance
- Cache operations are atomic and thread-safe

### Subprocess Safety
- Each subprocess runs in its own Python interpreter
- No shared state between processes
- Safe for high-concurrency scenarios

### Best Practices

1. **Cache Management**: Use separate cache directories for different document sets
2. **Configuration**: Keep configurations immutable during execution
3. **Error Handling**: Always handle both success and error responses
4. **Resource Management**: Limit concurrent processes to prevent resource exhaustion
5. **Timeouts**: Set appropriate timeouts for long-running queries

## Performance Considerations

### For Python Module Usage
- Corpus building is cached - first run will be slower
- BM25 index is cached and reused
- Consider using async interface for multiple queries

### For Subprocess Usage
- Each subprocess has initialization overhead
- Best for scenarios where you need process isolation
- Use batching for many small queries

### Memory Usage
- Corpus and indexes are loaded into memory
- Each subprocess uses ~100-500MB depending on document size
- Consider this when running many parallel processes

## Examples

See the `examples/` directory for complete working examples:
- `nodejs_integration.js`: Complete Node.js integration example
- Additional examples for other languages can be added

## Configuration Reference

For detailed configuration options, see the main `config.yaml` file and the `get_default_config()` function documentation.