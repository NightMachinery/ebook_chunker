import argparse
import logging
import sys
from pathlib import Path

from .constants import MAX_EBOOK_CHUNK_CHARS, MIN_EBOOK_CHUNK_CHARS


def setup_logging(verbosity: int) -> None:
    """Configure logging based on verbosity level."""
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG, 3: logging.DEBUG}

    level = levels.get(verbosity, logging.DEBUG)

    if verbosity >= 3:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    elif verbosity >= 2:
        format_str = "%(levelname)s: %(message)s"
    else:
        format_str = "%(message)s"

    logging.basicConfig(level=level, format=format_str, stream=sys.stderr)


def add_base_args(parser: argparse.ArgumentParser) -> None:
    """Add base arguments shared by CLIs."""
    parser.add_argument("epub_path", type=Path, help="Path to the EPUB file to process")

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
