# Code Quality Documentation

This directory contains comprehensive code quality analysis and improvement documentation for the lightweight-rag repository.

## Documents

### 📊 [CODE_QUALITY_REPORT.md](CODE_QUALITY_REPORT.md)
**Comprehensive Analysis Report**

A detailed 100+ page report covering:
- Cyclomatic complexity analysis
- Maintainability index measurements
- Test coverage analysis
- Code style and linting results
- Security vulnerability scan
- Function length and module size analysis
- Detailed recommendations by priority

**Key Findings:**
- ✅ Average complexity: 6.78 (Good - B grade)
- ✅ Security: 0 vulnerabilities 
- ✅ Maintainability: 16/18 modules rated 'A'
- ⚠️ Test coverage: 66% (target: 80%+)
- ⚠️ One critical function with complexity 137 (needs refactoring)
- ⚠️ Code style: 38 violations (down from 161)

**Overall Grade: B+** (Good with room for improvement)

---

### ✅ [CODE_QUALITY_CHECKLIST.md](CODE_QUALITY_CHECKLIST.md)
**Actionable Improvement Checklist**

A practical, step-by-step checklist organized by priority:

**Quick Wins** (Completed ✓)
- ✅ Code formatting (black, isort)
- ✅ Remove unused imports/variables
- ✅ Fix bare except clauses

**Critical Improvements** (In Progress)
- [ ] Refactor build_corpus function (F-137 → B-10)
- [ ] Improve test coverage (66% → 80%+)

**High Priority**
- [ ] Reduce complexity of C/D-rated functions
- [ ] Replace print statements with logging
- [ ] Fix remaining code style issues

**Medium Priority**
- [ ] Add type hints
- [ ] Improve documentation
- [ ] Add integration tests

**Estimated Timeline:** 3-4 weeks of focused work

---

### 🏗️ [ROADMAP.md](ROADMAP.md)
**Feature Development Roadmap**

The project's roadmap for future enhancements:
- ✅ Completed: Config system, modularization, performance improvements
- 🚧 In Progress: Retrieval quality improvements
- 📋 Planned: Configurable chunking, enhanced testing, API surface

---

## Quick Start

### Run Quality Checks

```bash
# Quick quality check
python scripts/check_quality.py

# Detailed analysis
python /tmp/code_quality_analysis.py

# Run tests with coverage
pytest --cov=lightweight_rag --cov-report=term-missing

# Check complexity
radon cc lightweight_rag/ -a -s

# Check maintainability
radon mi lightweight_rag/ -s

# Check code style
flake8 lightweight_rag/ --max-line-length=100 --max-complexity=10

# Security scan
bandit -r lightweight_rag/
```

### Apply Quick Fixes

```bash
# Format code
black lightweight_rag/ --line-length 100

# Organize imports
isort lightweight_rag/ --profile black

# Run tests
pytest tests/ -v
```

## Current Status

### Metrics Summary

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Test Coverage | 66% | 80%+ | 🔴 Below target |
| Avg Complexity | 6.78 (B) | ≤ 10 (B) | 🟢 On target |
| Maintainability | 16/18 'A' | 18/18 'A' | 🟡 Near target |
| Functions > C | 16 | 0 | 🔴 Needs work |
| Critical (F/D) | 2 | 0 | 🔴 Critical |
| Style Violations | 38 | < 10 | 🟡 Improved |
| Security Issues | 0 | 0 | 🟢 Perfect |

### Progress Tracking

**Completed:**
- ✅ Comprehensive code quality analysis
- ✅ Detailed documentation and reports
- ✅ Code formatting (black, isort)
- ✅ Removed unused imports and variables
- ✅ Reduced flake8 violations by 76% (161 → 38)

**In Progress:**
- 🔄 Refactoring high-complexity functions
- 🔄 Improving test coverage
- 🔄 Adding type hints

**Next Steps:**
1. Refactor `io_pdf.py::build_corpus` (complexity 137 → ≤10)
2. Add tests to increase coverage from 66% to 80%
3. Refactor remaining C-rated functions
4. Implement proper logging infrastructure

## Tools Used

- **radon**: Cyclomatic complexity and maintainability analysis
- **pytest + pytest-cov**: Testing and coverage measurement
- **flake8**: Code style and complexity linting
- **bandit**: Security vulnerability scanning
- **black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking (recommended)

## Best Practices

### Writing Bulletproof Code

1. **Keep Functions Small**
   - Target: < 50 lines per function
   - Complexity: ≤ 10 (grade B or better)

2. **Write Tests**
   - Aim for 80%+ coverage
   - Test edge cases and error paths
   - Use pytest fixtures for common setups

3. **Follow Style Guidelines**
   - Use black for formatting (line length: 100)
   - Use isort for imports
   - Follow PEP 8 conventions

4. **Document Your Code**
   - Add docstrings to all public functions
   - Include parameter types and return values
   - Add usage examples

5. **Handle Errors Gracefully**
   - Catch specific exceptions
   - Provide informative error messages
   - Use logging instead of print statements

6. **Review Regularly**
   - Run quality checks before committing
   - Review complexity metrics
   - Update tests with code changes

## Contributing

When contributing to this project:

1. **Before making changes:**
   - Run `python scripts/check_quality.py`
   - Note baseline metrics

2. **During development:**
   - Write tests for new functionality
   - Keep functions under 50 lines
   - Maintain complexity ≤ 10

3. **Before submitting:**
   - Run `black` and `isort`
   - Ensure all tests pass
   - Run `flake8` and fix violations
   - Update documentation

4. **After changes:**
   - Run quality checks again
   - Document improvements in PR
   - Update checklist progress

## References

- [Cyclomatic Complexity](https://en.wikipedia.org/wiki/Cyclomatic_complexity)
- [Maintainability Index](https://docs.microsoft.com/en-us/visualstudio/code-quality/code-metrics-values)
- [PEP 8 Style Guide](https://pep8.org/)
- [Python Testing Best Practices](https://docs.pytest.org/)
- [OWASP Security Practices](https://owasp.org/)

## Contact

For questions or suggestions about code quality:
- Open an issue on GitHub
- Reference the analysis reports in this directory
- Tag with `code-quality` label

---

**Last Updated:** 2024  
**Analysis Version:** 1.0  
**Repository:** https://github.com/socratic-irony/lightweight-rag
