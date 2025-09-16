# Plan: Structure-Aware Chunking for HTML/Markdown

## Problems with Current Approach
1. `semantic_chunking` uses sentence transformers and splits on sentence boundaries, ignoring HTML/Markdown structure
2. Could break in middle of HTML tags, code blocks, lists, etc.
3. Converts everything to plain text first, losing formatting

## New Approach: Structure-Aware Chunking

### Core Strategy
- Split content at structural boundaries (paragraphs, headings, list items)
- Maintain balanced chunk sizes (roughly equal)
- Preserve HTML/Markdown integrity

### Implementation Plan

1. **Create new `structure_aware_chunking` function** in `epub_chunker.py`:
   - Parse HTML to identify safe split points
   - Find optimal split point near the middle that preserves structure
   - Support recursive splitting for chunks still too large

2. **Safe split points hierarchy** (in order of preference):
   - Between top-level sections (`<h1>`, `<h2>`, etc.)
   - Between paragraphs (`</p>`)
   - Between list items (`</li>`)
   - Between block elements (`</div>`, `</blockquote>`)
   - Between sentences (fallback)

3. **Update `chunk_epub` function**:
   - Add `format` parameter (default "md")
   - Keep HTML structure intact initially
   - Convert to target format AFTER chunking:
     - `html`: Keep original HTML
     - `md`: Convert HTML to Markdown via pandoc
     - `txt`: Extract plain text

4. **Replace `semantic_chunking` calls**:
   - Use new `structure_aware_chunking` function
   - Pass HTML content instead of plain text

5. **Update CLI** (`cli.py`):
   - Pass `format` parameter to `chunk_epub`
   - Remove redundant format conversion loop (lines 280-288)

## Benefits
- No broken HTML tags or markdown syntax
- Better preservation of document structure
- More predictable chunk boundaries
- Balanced chunk sizes
- Format conversion happens once with full context