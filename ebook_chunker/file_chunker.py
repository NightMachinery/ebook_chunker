import logging
from pathlib import Path
from typing import List, Optional, Callable
from enum import Enum

from bs4 import BeautifulSoup

from .pandoc_utils import convert_text
from .epub_chunker import (
    epub_to_segments,
    _chunk_segments,
    _html_segments_from_soup,
    _extract_text_from_html,
    _safe_html_join,
)


class FileBoundary(Enum):
    NONE = "none"
    SOFT = "soft"
    HARD = "hard"


def detect_file_format(file_path: Path) -> str:
    """Detect file format based on extension."""
    markdown_format = "markdown-yaml_metadata_block"

    suffix = file_path.suffix.lower()
    format_mapping = {
        ".epub": "epub",
        ".html": "html",
        ".htm": "html",
        ".md": markdown_format,
        ".markdown": markdown_format,
        ".txt": "plain",
        ".text": "plain",
        ".docx": "docx",
        ".doc": "doc",
        ".pdf": "pdf",
        ".odt": "odt",
        ".rtf": "rtf",
    }
    return format_mapping.get(suffix, "plain")


def file_to_segments(
    file_path: Path, *, skip_index_p: bool = True
) -> List[tuple[str, str]]:
    """Extract segments from any file type."""
    logger = logging.getLogger(__name__)
    file_format = detect_file_format(file_path)

    if file_format == "epub":
        # Use EPUB-specific segment extraction
        return epub_to_segments(str(file_path), skip_index_p=skip_index_p)

    elif file_format == "html":
        # Parse HTML directly
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, "html.parser")
        return _html_segments_from_soup(soup)

    else:
        # Convert other formats to HTML, then parse
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            # Map plain to markdown for pandoc compatibility
            pandoc_format = "markdown" if file_format == "plain" else file_format
            html_content = convert_text(
                content, from_format=pandoc_format, to_format="html"
            )
        except Exception as e:
            logger.warning(f"Pandoc conversion failed for {file_path}: {e}")
            # Fallback: wrap plain text in HTML paragraphs
            lines = content.split("\n")
            html_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    # Basic HTML escaping
                    line = (
                        line.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    html_lines.append(f"<p>{line}</p>")
                else:
                    html_lines.append("<br>")
            html_content = "\n".join(html_lines)

        # Parse the HTML to extract segments
        soup = BeautifulSoup(html_content, "html.parser")
        return _html_segments_from_soup(soup)


def chunk_files(
    file_paths: List[Path],
    *,
    max_chunk_chars: int,
    min_chunk_chars: int,
    file_boundary: FileBoundary = FileBoundary.SOFT,
    format: str = "md",
    converter: Optional[Callable[[str], str]] = None,
    skip_index_p: bool = True,
) -> List[str]:
    """
    Chunk multiple files with specified boundary handling.

    Args:
        file_paths: List of input file paths
        max_chunk_chars: Maximum characters per chunk
        min_chunk_chars: Minimum characters per chunk
        file_boundary: How to handle boundaries between files
        format: Output format ('md', 'txt', 'html')
        converter: Optional converter function
        skip_index_p: Skip 'Index' sections in EPUB files

    Returns:
        List of chunked content strings
    """
    logger = logging.getLogger(__name__)

    if not file_paths:
        return []

    logger.info(
        f"Processing {len(file_paths)} files with boundary={file_boundary.value}"
    )

    # Extract segments from all files
    all_segments: List[tuple[str, str]] = []

    try:
        boundary_marker = {
            FileBoundary.HARD: "hard_file_boundary",
            FileBoundary.SOFT: "section_break",
            FileBoundary.NONE: None,
        }[file_boundary]
    except KeyError as err:
        raise ValueError(f"Unsupported file boundary: {file_boundary}") from err

    for i, file_path in enumerate(file_paths):
        logger.debug(f"Processing file {i+1}/{len(file_paths)}: {file_path}")

        try:
            # Extract segments directly from the file
            file_segments = file_to_segments(file_path, skip_index_p=skip_index_p)

            if not file_segments:
                logger.warning(f"No segments extracted from file: {file_path}")
                continue

            all_segments.extend(file_segments)

            if boundary_marker and i < len(file_paths) - 1:
                all_segments.append(("", boundary_marker))

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue

    if not all_segments:
        logger.warning("No content segments extracted from any files")
        return []

    # Apply chunking (unified function handles all boundary types)
    html_chunks = _chunk_segments(
        all_segments,
        max_chunk_chars=max_chunk_chars,
        min_chunk_chars=min_chunk_chars,
    )

    # Convert to target format
    if format == "html":
        return html_chunks

    def default_convert(html: str, to_format: str) -> str:
        if to_format == "html":
            return html
        return convert_text(html, from_format="html", to_format=to_format)

    if format not in {"md", "markdown", "txt", "text", "plain", "html"}:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Converting {len(html_chunks)} chunks to {format} format")
    out_chunks: List[str] = []
    for html in html_chunks:
        if converter is not None:
            out_chunks.append(converter(html))
        else:
            out_chunks.append(default_convert(html, format))

    return out_chunks
