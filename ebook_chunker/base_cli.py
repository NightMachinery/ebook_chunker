import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Tuple

from .constants import MAX_EBOOK_CHUNK_CHARS, MIN_EBOOK_CHUNK_CHARS


def _normalize_names(
    names: Iterable[str], *, fallback: Tuple[str, ...]
) -> Tuple[str, ...]:
    cleaned = [name for name in names if name]
    unique = tuple(dict.fromkeys(cleaned))
    if not unique:
        return fallback
    return unique


def setup_logging(
    verbosity: int,
    *,
    package_names: Iterable[str] = (
        "ebook_chunker",
        "ebook_feeder",
        "epub_sum_lib",
    ),
    muted_packages: Iterable[str] = ("LiteLLM", "litellm"),
) -> None:
    """Configure logging with verbose output scoped to project packages."""

    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG, 3: logging.DEBUG}
    level = levels.get(verbosity, logging.DEBUG)

    if verbosity >= 3:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    elif verbosity >= 2:
        format_str = "%(levelname)s: %(message)s"
    else:
        format_str = "%(message)s"

    normalized = _normalize_names(package_names, fallback=("ebook_chunker",))
    muted = _normalize_names(muted_packages, fallback=())

    formatter = logging.Formatter(format_str)

    root_handler = logging.StreamHandler(sys.stderr)
    root_handler.setFormatter(formatter)
    root_handler.setLevel(logging.WARNING)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)
    root_logger.addHandler(root_handler)

    project_handler = logging.StreamHandler(sys.stderr)
    project_handler.setFormatter(formatter)
    project_handler.setLevel(level)

    for package_name in normalized:
        package_logger = logging.getLogger(package_name)
        package_logger.handlers.clear()
        package_logger.setLevel(level)
        package_logger.addHandler(project_handler)
        package_logger.propagate = False

    for muted_name in muted:
        muted_logger = logging.getLogger(muted_name)
        muted_logger.setLevel(logging.WARNING)


def add_base_args(parser: argparse.ArgumentParser) -> None:
    """Add base arguments shared by CLIs."""
    parser.add_argument(
        "input_files",
        type=Path,
        nargs="+",
        help="Path(s) to input file(s) to process (EPUB, HTML, Markdown, Text, etc.)",
    )

    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=MAX_EBOOK_CHUNK_CHARS,
        help=f"Maximum characters per chunk (default: {MAX_EBOOK_CHUNK_CHARS})",
    )

    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=MIN_EBOOK_CHUNK_CHARS,
        help=f"Minimum characters per chunk (default: {MIN_EBOOK_CHUNK_CHARS})",
    )

    parser.add_argument(
        "--skip-index",
        dest="skip_index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop at 'Index' heading in last section",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times: -v, -vv, -vvv)",
    )

    # Intentionally do not add common execution flags such as --dry-run or --overwrite here.
    # Those may vary per command and are declared in each CLI.
