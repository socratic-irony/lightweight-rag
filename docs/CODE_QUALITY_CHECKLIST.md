# Code Quality Improvement Checklist

This document provides an actionable checklist for improving code quality based on the comprehensive analysis report.

## Quick Wins (Can be done immediately)

### Code Style Fixes
- [ ] Run `black lightweight_rag/ --line-length 100` to auto-format code
- [ ] Run `isort lightweight_rag/ --profile black` to organize imports
- [ ] Remove unused import: `collections.Counter` from rerank.py
- [ ] Remove unused variable `q` from rerank.py:77
- [ ] Add newlines at end of files: prf.py, rerank.py, scoring.py

### Documentation
- [ ] Add docstrings to functions missing them
- [ ] Update README with code quality badges
- [ ] Document logging configuration
- [ ] Add type hints to public API functions

## Critical Improvements

### 1. Refactor build_corpus Function (CRITICAL)
**Priority:** HIGHEST
**Complexity:** F (137) → Target: B (≤10)
**File:** lightweight_rag/io_pdf.py

Break down into smaller functions:
- [ ] Extract `_discover_pdf_files(pdf_dir)` function
- [ ] Extract `_try_load_cache(files, cache_dir)` function  
- [ ] Extract `_extract_and_chunk_pdfs(files, config)` function
- [ ] Extract `_enrich_citations_parallel(corpus, config)` function
- [ ] Extract `_save_corpus_cache(corpus, files, cache_dir)` function
- [ ] Update tests to cover new functions
- [ ] Verify complexity drops to acceptable level

### 2. Improve Test Coverage (66% → 80%+)
**Priority:** HIGH

#### io_pdf.py Coverage (56% → 80%)
- [ ] Test malformed PDF handling
- [ ] Test various text encoding scenarios
- [ ] Test different chunking configurations
- [ ] Test cache invalidation
- [ ] Test text quality edge cases
- [ ] Test sliding window boundaries
- [ ] Test concurrent PDF processing

#### cite.py Coverage (56% → 80%)
- [ ] Test Crossref API timeout handling
- [ ] Test OpenAlex API error responses
- [ ] Test invalid DOI formats
- [ ] Test batch processing edge cases
- [ ] Test cache expiration logic
- [ ] Test metadata enrichment failures

#### diversity.py Coverage (50% → 80%)
- [ ] Test MMR selection with various thresholds
- [ ] Test with empty result sets
- [ ] Test with single result
- [ ] Test cosine similarity edge cases
- [ ] Test TF-IDF vector calculation
- [ ] Test format_results variations

#### environment.py Coverage (51% → 80%)
- [ ] Test environment detection in different contexts
- [ ] Test path resolution scenarios
- [ ] Test config adaptation logic
- [ ] Test subprocess detection

## High Priority Improvements

### 3. Reduce Function Complexity (C/D → B)

#### io_pdf.py::create_sliding_windows (D-22 → B-9)
- [ ] Extract boundary detection logic
- [ ] Extract sentence splitting logic
- [ ] Simplify window creation loop
- [ ] Add helper function for overlap calculation

#### diversity.py::mmr_selection (C-19 → B-9)
- [ ] Extract similarity calculation
- [ ] Extract candidate scoring logic
- [ ] Simplify selection loop
- [ ] Add helper for diversity threshold check

#### main.py::search_topk (C-19 → B-9)
- [ ] Extract BM25 scoring phase
- [ ] Extract bonus calculation phase
- [ ] Extract reranking phase
- [ ] Extract diversity selection phase
- [ ] Create pipeline orchestrator pattern

#### cite.py::openalex_meta_for_doi (C-18 → B-8)
- [ ] Extract field extraction logic
- [ ] Extract author parsing
- [ ] Extract location/URL extraction
- [ ] Add helper functions for nested data

#### subprocess_interface.py::validate_input (C-18 → B-8)
- [ ] Consider using JSON schema validation library
- [ ] Extract query validation
- [ ] Extract config validation
- [ ] Extract config_file validation

#### cli_subprocess.py::batch_processing_mode (C-20 → B-9)
- [ ] Extract batch reading logic
- [ ] Extract result writing logic
- [ ] Extract error handling logic
- [ ] Simplify main loop

### 4. Replace Print Statements with Logging

- [ ] Add logging configuration to config.yaml
- [ ] Create logging.py module with logger setup
- [ ] Replace prints in io_pdf.py with logger calls
- [ ] Replace prints in cite.py with logger calls
- [ ] Replace prints in main.py with logger calls
- [ ] Replace prints in index.py with logger calls
- [ ] Replace prints in cli modules with logger calls
- [ ] Add log levels (DEBUG, INFO, WARNING, ERROR)
- [ ] Add structured logging for key events
- [ ] Update documentation for logging configuration

## Medium Priority Improvements

### 5. Add Type Hints

- [ ] Add type hints to config.py public functions
- [ ] Add type hints to io_pdf.py public functions
- [ ] Add type hints to index.py public functions
- [ ] Add type hints to scoring.py functions
- [ ] Add type hints to cite.py public functions
- [ ] Add type hints to main.py functions
- [ ] Set up mypy configuration
- [ ] Run mypy and fix type issues
- [ ] Add type hints to test files

### 6. Improve Documentation

- [ ] Add comprehensive docstrings to all public functions
- [ ] Document function parameters with types
- [ ] Document return values with types
- [ ] Add usage examples in docstrings
- [ ] Document exceptions that can be raised
- [ ] Add module-level documentation
- [ ] Create architecture documentation
- [ ] Add contributing guidelines

### 7. Integration Testing

- [ ] Create fixtures directory with sample PDFs
- [ ] Add end-to-end pipeline test
- [ ] Add configuration variation tests
- [ ] Add error recovery tests
- [ ] Add cache invalidation integration tests
- [ ] Add concurrent operation tests
- [ ] Add performance regression tests

## Low Priority (Nice to Have)

### 8. Performance Optimization

- [ ] Add pytest-benchmark for performance tracking
- [ ] Profile critical paths (PDF extraction, indexing, search)
- [ ] Optimize hot spots identified in profiling
- [ ] Add performance tests to CI/CD
- [ ] Document performance characteristics
- [ ] Consider connection pooling for HTTP clients
- [ ] Consider lazy loading for large models

### 9. Enhanced Tooling

- [ ] Add pre-commit hooks for formatting
- [ ] Add GitHub Actions for code quality checks
- [ ] Add coverage reporting to CI/CD
- [ ] Add complexity checking to CI/CD
- [ ] Set up automatic dependency updates
- [ ] Add security scanning to CI/CD

### 10. Documentation Website

- [ ] Set up Sphinx or MkDocs
- [ ] Generate API documentation
- [ ] Create user guide
- [ ] Add architecture diagrams
- [ ] Add contribution guidelines
- [ ] Add deployment guide
- [ ] Host on GitHub Pages

## Verification Steps

After completing improvements:

- [ ] Run `pytest tests/ --cov=lightweight_rag --cov-report=term-missing`
  - Target: >= 80% coverage
  
- [ ] Run `radon cc lightweight_rag/ -a -s`
  - Target: Average complexity <= 8
  - No functions with complexity > 10
  
- [ ] Run `radon mi lightweight_rag/ -s`
  - Target: All modules rated 'A'
  
- [ ] Run `flake8 lightweight_rag/ --max-line-length=100 --max-complexity=10`
  - Target: Zero violations
  
- [ ] Run `bandit -r lightweight_rag/`
  - Target: No security issues
  
- [ ] Run `mypy lightweight_rag/`
  - Target: No type errors
  
- [ ] Run full test suite
  - Target: All tests passing
  
- [ ] Manual testing of key features
  - PDF extraction
  - Search functionality
  - Citation enrichment
  - CLI interface
  
- [ ] Update documentation
  - README
  - API docs
  - Configuration guide
  
- [ ] Create summary report
  - Before/after metrics
  - Improvements achieved
  - Remaining work

## Success Metrics

| Metric | Before | Target | Status |
|--------|--------|--------|--------|
| Test Coverage | 66% | 80%+ | 🔴 |
| Avg Complexity | 6.78 | <= 8 | 🟢 |
| Functions > C | 16 | 0 | 🔴 |
| Critical (F/D) | 2 | 0 | 🔴 |
| Style Violations | 161 | < 10 | 🔴 |
| Security Issues | 0 | 0 | 🟢 |
| Maintainability | 16/18 'A' | 18/18 'A' | 🟡 |
| Documentation | Good | Excellent | 🟡 |

## Notes

- Focus on critical items first (build_corpus refactor, test coverage)
- Quick wins (formatting, removing unused imports) can be done anytime
- Type hints and documentation can be added incrementally
- Performance optimization should be data-driven (profile first)
- Keep all changes backwards compatible
- Update tests alongside code changes
- Document all significant changes

## Timeline Estimate

- **Week 1:** Critical fixes (build_corpus, code style, critical tests)
- **Week 2:** High priority (complexity reduction, logging, more tests)
- **Week 3:** Medium priority (type hints, documentation, integration tests)
- **Week 4:** Polish (performance, tooling, final verification)

**Total Effort:** 3-4 weeks of focused work
