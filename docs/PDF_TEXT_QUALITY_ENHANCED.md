# PDF Text Quality Improvements

This document describes the text quality improvements added to handle PDFs with encoding issues or garbled text extraction.

## Problem

Some PDFs contain text that extracts as control characters or garbled content, such as:
```
\u0000\u0001\u0002\u0003\u0001\u0005\u0006\u0007\b\t \u0005\u000b\u0001\u0005\f\b\t\u0005\r\u000e\u000f
```

This happens due to:
- Font encoding issues in the PDF
- Encrypted or protected text
- Scanned documents with poor OCR
- Corrupted PDF files

## Solution

Added comprehensive text quality validation and cleaning in `io_pdf.py`:

### 1. Text Quality Validation (`is_text_quality_good`)

Checks text for:
- **Control characters**: Rejects text with >5% control characters (excluding tabs, newlines)
- **Printable ratio**: Requires ≥70% printable characters
- **Repeated patterns**: Detects excessive character repetition (encoding artifacts)
- **Character distribution**: Ensures presence of common letters

### 2. Enhanced Text Cleaning (`clean_text` + `normalize_text`)

- **Control character removal**: Removes null bytes and problematic control characters
- **Soft hyphen handling**: Removes soft hyphens (U+00AD) completely
- **Hard hyphen normalization**: Reconnects words split by line-break hyphens (`word-\nbreak` → `wordbreak`)
- **Line break normalization**: Converts newlines to spaces
- **Unicode normalization**: NFKC normalization for consistent character representation
- **Whitespace cleanup**: Normalizes excessive whitespace to single spaces

### 3. Enhanced PDF Extraction (`extract_pdf_pages`)

Multi-stage extraction process:
1. **Primary extraction**: Standard `page.get_text()`
2. **Quality check**: Validate and clean extracted text
3. **Fallback methods**: If quality is poor, try:
   - TextPage extraction with different options
   - Block-based extraction
4. **Warning system**: Mark problematic pages with `[TEXT_QUALITY_WARNING]`
5. **Filtering**: Skip pages that fail all quality checks

### 4. Configuration Options

Added to `config.yaml`:
```yaml
indexing:
  text_quality_check: true       # Enable text quality validation
  min_readable_ratio: 0.7        # Minimum ratio of readable characters
```

## Text Normalization Examples

The enhanced normalization handles common PDF extraction issues:

| Issue | Input | Output |
|-------|-------|---------|
| Soft hyphens | `re\u00ADsearch` | `research` |
| Line-break hyphens | `hyphen-\nated` | `hyphenated` |
| Mixed issues | `de-\nhy\u00ADphen-\nated   text` | `dehyphenated text` |

## Results

- Garbled/corrupted pages are automatically filtered out
- Clean, readable text is preserved with proper word reconstruction
- Multiple extraction methods increase success rate
- Warnings help identify problematic source documents

## Testing

Run the enhanced normalization test:
```bash
python3 test_enhanced_normalization.py
```

This validates both quality detection and text normalization against various samples including hyphenation, soft hyphens, and the original problematic case.

## Impact

- Improved search accuracy by removing noise and reconstructing words
- Better citation extraction from properly normalized text
- Reduced false matches from garbled content
- Enhanced overall corpus quality with readable, searchable text