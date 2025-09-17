from pynight.common_icecream import (
    ic,
)  #: used for debugging, DO NOT REMOVE even if currently unused

from bs4 import XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from bs4 import BeautifulSoup, Tag

import io
import re
import logging
from typing import Callable, Iterable, List, Optional, Tuple
from epub_sum_lib.epubsplit import SplitEpub
from .constants import MAX_EBOOK_CHUNK_CHARS, MIN_EBOOK_CHUNK_CHARS


def _extract_text_from_html(html_content: str) -> str:
    """Extracts clean text from an HTML string."""
    soup = BeautifulSoup(html_content, "html.parser")
    return re.sub(r"\n\s*\n", "\n\n", soup.get_text(separator="\n", strip=True))


def _split_sentences(text: str) -> List[str]:
    """Split plain text into sentences using a simple regex."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences


def _wrap_as_html_paragraph(text: str) -> str:
    return f"<p>{text}</p>"


def _safe_html_join(parts: Iterable[str]) -> str:
    """Join HTML parts with a newline, wrapped in a container for validity."""
    joined = "\n".join(part for part in parts if part and part.strip())
    if not joined.strip():
        return ""

    # return f"<div>\n{joined}\n</div>"
    return joined


def _html_segments_from_soup(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    """
    Produce a flat list of (html, role) segments from a soup.
    role is one of: 'heading', 'list_item', 'block', 'other'.
    """
    root = soup.body if soup.body else soup
    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    list_containers = {"ul", "ol"}
    block_tags = {"p", "blockquote", "pre", "table", "figure", "hr"}
    container_tags = {"div", "section", "article", "main", "header", "footer", "aside"}

    segments: List[Tuple[str, str]] = []

    def add_segments(node: Tag) -> None:
        if not isinstance(node, Tag):
            return
        name = node.name.lower() if node.name else ""

        if name in heading_tags:
            segments.append((str(node), "heading"))
            return
        if name in list_containers:
            for li in node.find_all("li", recursive=False):
                segments.append((str(li), "list_item"))
            return
        if name in block_tags:
            segments.append((str(node), "block"))
            return
        if name in container_tags:
            # Recurse into children to find more granular blocks
            before_len = len(segments)
            for child in list(node.children):
                if isinstance(child, Tag):
                    add_segments(child)
            # If no segments were added, keep the container as a single block
            if len(segments) == before_len:
                segments.append((str(node), "block"))
            return
        # Default: keep as block
        segments.append((str(node), "block"))

    for child in list(root.children):
        if isinstance(child, Tag):
            add_segments(child)

    if not segments:
        html = str(root)
        if html.strip():
            segments.append((html, "other"))

    return segments


def _fallback_split_html(
    html: str, *, max_chunk_chars: int, min_chunk_chars: int
) -> List[str]:
    """
    Fallback splitter for oversized HTML segments.
    Converts to plain text, splits by sentences between size bounds,
    and wraps each chunk in simple paragraph HTML.
    """
    logger = logging.getLogger(__name__)
    logger.warning(
        "Using fallback sentence-based splitter; this loses HTML structure and may drop formatting."
    )
    text = _extract_text_from_html(html)
    if len(text) <= max_chunk_chars:
        return [html]

    sentences = _split_sentences(text)
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for s in sentences:
        buf.append(s)
        size += len(s)
        if size >= max_chunk_chars and size >= min_chunk_chars:
            chunks.append(_wrap_as_html_paragraph(" ".join(buf)))
            buf, size = [], 0
    if buf:
        if (
            chunks
            and (len(_extract_text_from_html(chunks[-1])) + size) < min_chunk_chars
        ):
            prev = _extract_text_from_html(chunks[-1]) + " " + " ".join(buf)
            chunks[-1] = _wrap_as_html_paragraph(prev)
        else:
            chunks.append(_wrap_as_html_paragraph(" ".join(buf)))
    return chunks


def _chunk_segments(
    segments: List[Tuple[str, str]],
    *,
    max_chunk_chars: int,
    min_chunk_chars: int,
) -> List[str]:
    """Chunk a precomputed list of (html, role) segments.

    Special role 'section_break' is treated as a soft boundary: if the
    current chunk has at least min_chunk_chars, we flush before starting
    the next section; otherwise we continue accumulating.
    """
    logger = logging.getLogger(__name__)

    def seg_len(seg_html: str) -> int:
        return len(_extract_text_from_html(seg_html))

    chunks: List[str] = []
    current: List[Tuple[str, str]] = []  # (html, role)
    current_len = 0
    last_preferred_boundary: Optional[int] = None

    for seg_html, role in segments:
        # ic(role)

        # Soft finalize before new section if current meets min
        if role == "section_break":
            if current and current_len >= min_chunk_chars:
                logger.debug(
                    "flush_on_section_boundary: current_len=%d chunks_so_far=%d",
                    current_len,
                    len(chunks),
                )
                # current.append((_wrap_as_html_paragraph("Section Break"), None))  #: for debugging

                chunks.append(_safe_html_join([h for h, _ in current]))
                current, current_len, last_preferred_boundary = [], 0, None
            # Do not add a segment for section_break; continue
            continue

        # @disabled Start new chunk at headings when current chunk is strong enough
        if False and role == "heading" and current and current_len >= min_chunk_chars:
            logger.debug(
                "flush_on_heading: current_len=%d chunks_so_far=%d",
                current_len,
                len(chunks),
            )
            chunks.append(_safe_html_join([h for h, _ in current]))
            current, current_len, last_preferred_boundary = [], 0, None

        current.append((seg_html, role))
        current_seg_len = seg_len(seg_html)
        current_len += current_seg_len
        if role in {
            "block",
            "list_item",
            "heading",
            "section_break",
        }:
            last_preferred_boundary = len(current)

        if current_len >= max_chunk_chars:
            split_at: Optional[int] = None
            if last_preferred_boundary is not None and last_preferred_boundary > 0:
                size_until = sum(
                    seg_len(h) for h, _ in current[:last_preferred_boundary]
                )
                if size_until >= min_chunk_chars:
                    split_at = last_preferred_boundary
            if split_at is None and (current_len - current_seg_len) >= min_chunk_chars:
                split_at = len(current)

            if split_at is not None:
                boundary_kind = (
                    "preferred_boundary"
                    if split_at == (last_preferred_boundary or -1)
                    else "end_boundary"
                )
                piece = current[:split_at]
                remaining = current[split_at:]
                piece_html = _safe_html_join([h for h, _ in piece])
                piece_text_len = len(_extract_text_from_html(piece_html))
                logger.debug(
                    "split: kind=%s split_at=%d piece_text_len=%d remaining_segments=%d",
                    boundary_kind,
                    split_at,
                    piece_text_len,
                    len(remaining),
                )
                if piece_text_len > max_chunk_chars:
                    fb = _fallback_split_html(
                        piece_html,
                        max_chunk_chars=max_chunk_chars,
                        min_chunk_chars=min_chunk_chars,
                    )
                    logger.debug("fallback_piece_used: produced=%d", len(fb))
                    chunks.extend(fb)
                else:
                    chunks.append(piece_html)
                current = remaining
                current_len = sum(seg_len(h) for h, _ in current)
                last_preferred_boundary = None

    if current:
        final_html = _safe_html_join([h for h, _ in current])
        final_len = len(_extract_text_from_html(final_html))
        logger.debug("final_piece: text_len=%d", final_len)
        if final_len > max_chunk_chars:
            fb = _fallback_split_html(
                final_html,
                max_chunk_chars=max_chunk_chars,
                min_chunk_chars=min_chunk_chars,
            )
            logger.debug("fallback_final_used: produced=%d", len(fb))
            chunks.extend(fb)
        else:
            chunks.append(final_html)

    # Merge very small trailing chunk into previous when possible
    merged: List[str] = []
    pre_merge_count = len(chunks)
    for i, ch in enumerate(chunks):
        #: only merge the very last chunk
        if merged and i == (len(chunks) - 1):
            prev_text_len = len(_extract_text_from_html(merged[-1]))
            cur_text_len = len(_extract_text_from_html(ch))
            if (
                cur_text_len < min_chunk_chars
                and prev_text_len + cur_text_len <= max_chunk_chars
            ):
                logger.debug(
                    "merge_small_tail: prev_len=%d cur_len=%d",
                    prev_text_len,
                    cur_text_len,
                )
                merged[-1] = _safe_html_join([merged[-1], ch])
                continue
        merged.append(ch)
    logger.debug("result: pre_merge=%d post_merge=%d", pre_merge_count, len(merged))
    return merged


def structure_aware_chunking(
    html: str,
    *,
    max_chunk_chars: int,
    min_chunk_chars: int,
) -> List[str]:
    """
    Split HTML by structural boundaries while balancing chunk sizes.

    Returns a list of HTML chunks that do not break tags.
    """
    logger = logging.getLogger(__name__)
    soup = BeautifulSoup(html, "html.parser")
    segments = _html_segments_from_soup(soup)

    if logger.isEnabledFor(logging.DEBUG):
        total_text_len = len(_extract_text_from_html(html))
        role_counts: dict[str, int] = {}
        for _, role in segments:
            role_counts[role] = role_counts.get(role, 0) + 1
        logger.debug(
            "structure_aware_chunking: text_len=%d segments=%d roles=%s bounds(min=%d,max=%d)",
            total_text_len,
            len(segments),
            role_counts,
            min_chunk_chars,
            max_chunk_chars,
        )

    return _chunk_segments(
        segments, max_chunk_chars=max_chunk_chars, min_chunk_chars=min_chunk_chars
    )


def chunk_epub(
    epub_path: str,
    *,
    max_chunk_chars: int = MAX_EBOOK_CHUNK_CHARS,
    min_chunk_chars: int = MIN_EBOOK_CHUNK_CHARS,
    format: str = "md",
    converter: Optional[Callable[[str], str]] = None,
    skip_index_p: bool = True,
) -> list[str]:
    """
    Chunk an EPUB with structure-aware splitting and optional format conversion.

    epub_path: Path to the EPUB file.
    format: One of 'md', 'txt', 'html'.
    converter: Optional callable that converts HTML to target format.
               Signature: converter(html: str) -> str
               If not provided, will attempt to use pandoc if available,
               otherwise fall back to plain-text extraction for non-HTML.
    skip_index_p: If True, in the last section, encountering a heading titled
                  'Index' (case-insensitive) ends processing at that point.
    """
    logger = logging.getLogger(__name__)
    html_chunks: List[str] = []

    with open(epub_path, "rb") as f:
        epub_io = io.BytesIO(f.read())

    splitter = SplitEpub(epub_io)
    lines = splitter.get_split_lines()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("split_lines: count=%d", len(lines))

    # Identify sections by TOC lines
    sections: List[List[int]] = []
    current_section_lines: List[int] = []
    for i, line in enumerate(lines):
        if line.get("toc"):
            if current_section_lines:
                sections.append(current_section_lines)
            current_section_lines = [i]
        elif current_section_lines:
            current_section_lines.append(i)
    if current_section_lines:
        sections.append(current_section_lines)
    if not sections:
        sections.append(list(range(len(lines))))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("sections: count=%d", len(sections))

    # Build a single segment list across sections, inserting soft section boundaries
    combined_segments: List[Tuple[str, str]] = []
    for sec_idx, section_linenums in enumerate(sections):
        files_in_section = splitter.get_split_files(section_linenums)
        section_html_parts: List[str] = []
        for _, _, _, filedata in files_in_section:
            try:
                soup = BeautifulSoup(filedata, "html.parser")
                body = soup.body if soup.body else soup
                inner_html = "".join(str(child) for child in body.children)
                if not inner_html.strip():
                    inner_html = str(body)
                section_html_parts.append(inner_html)
            except Exception:
                section_html_parts.append(filedata)
        section_html = _safe_html_join(section_html_parts)
        if not section_html.strip():
            continue
        if logger.isEnabledFor(logging.DEBUG):
            section_text_len = len(_extract_text_from_html(section_html))
            logger.debug(
                "section_%d: files=%d text_len=%d",
                sec_idx,
                len(section_html_parts),
                section_text_len,
            )
        # Parse section html into structural segments and append
        section_soup = BeautifulSoup(section_html, "html.parser")
        segs = _html_segments_from_soup(section_soup)

        # If this is the last section and skip_index_p is set, cut at an 'Index' heading
        if skip_index_p and sec_idx <= len(sections) - 2:
            #: Sometimes it's not necessary in the very last section, so have some tolerance.

            stop_at = None
            for i, (seg_html, role) in enumerate(segs):
                if role == "heading":
                    heading_text = _extract_text_from_html(seg_html).strip().lower()
                    # ic(heading_text)

                    if heading_text in [
                        "index",
                        "acknowledgments",
                    ]:
                        stop_at = i
                        break
            if stop_at is not None:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "skip_index: last_section=%d stop_at_segment=%d",
                        sec_idx,
                        stop_at,
                    )
                segs = segs[:stop_at]

        combined_segments.extend(segs)
        # Insert a soft section boundary between sections
        if sec_idx < len(sections) - 1:
            combined_segments.append(
                (
                    "",
                    "section_break",
                )
            )

    if logger.isEnabledFor(logging.DEBUG):
        roles_summary: dict[str, int] = {}
        for _, r in combined_segments:
            roles_summary[r] = roles_summary.get(r, 0) + 1
        total_text_len = sum(
            len(_extract_text_from_html(h))
            for h, r in combined_segments
            if r != "section_break"
        )
        logger.debug(
            "combined_segments: segments=%d roles=%s total_text_len=%d",
            len(combined_segments),
            roles_summary,
            total_text_len,
        )

    html_chunks = _chunk_segments(
        combined_segments,
        max_chunk_chars=max_chunk_chars,
        min_chunk_chars=min_chunk_chars,
    )

    def default_convert(html: str, to_format: str) -> str:
        if to_format == "html":
            return html
        from .pandoc_utils import convert_text as _pandoc_convert

        return _pandoc_convert(html, from_format="html", to_format=to_format)

    if format not in {"md", "markdown", "txt", "text", "plain", "html"}:
        raise ValueError(f"Unsupported format: {format}")

    if format in {"html"}:
        return html_chunks

    out_chunks: List[str] = []
    for html in html_chunks:
        if converter is not None:
            out_chunks.append(converter(html))
        else:
            out_chunks.append(default_convert(html, format))
    return out_chunks
