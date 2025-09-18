import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Union

from .file_chunker import chunk_files, FileBoundary
from .pandoc_utils import get_file_extension
from .constants import MAX_EBOOK_CHUNK_CHARS, MIN_EBOOK_CHUNK_CHARS
from .base_cli import add_base_args, setup_logging


def calculate_padding(num_chunks: int, pad_option: Union[str, int]) -> Optional[int]:
    """
    Calculate padding width for chunk numbering.

    Args:
        num_chunks: Total number of chunks
        pad_option: 'auto', 'none', or integer value

    Returns:
        Padding width or None for no padding
    """
    if pad_option == "none":
        return None
    elif pad_option == "auto":
        return len(str(num_chunks))
    else:
        try:
            return int(pad_option)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid padding option: {pad_option}")


def generate_filename(
    chunk_index: int,
    padding: Optional[int],
    output_pattern: str,
    format_name: str,
    total_chunks: int,
) -> Path:
    """Generate filename for a chunk."""
    if padding is None:
        chunk_num = str(chunk_index + 1)
    else:
        chunk_num = str(chunk_index + 1).zfill(padding)

    extension = get_file_extension(format_name)

    # Parse the output pattern
    path = Path(output_pattern)

    # Replace placeholders
    filename_template = (
        path.name if path.name else f"{{input_stem}}_{{chunk_num:0{padding}d}}.{{ext}}"
    )

    # If path has no name part, create default pattern
    if not path.name:
        path = path / f"{{input_stem}}_chunks"
        filename_template = f"{{input_stem}}_{{chunk_num}}.{{ext}}"

    # Format the filename
    formatted_name = filename_template.format(
        chunk_num=chunk_num,
        ext=extension,
        input_stem="{input_stem}",  # Will be replaced later with actual stem
        total_chunks=total_chunks,
    )

    return path.parent / formatted_name


def print_chunk_preview(chunk: str, filename: Path, chunk_index: int) -> None:
    """Print a preview of the chunk for dry-run mode."""
    lines = chunk.split("\n")
    preview_lines = lines[:5]  # First 5 lines

    print(f"--- Chunk {chunk_index + 1} ---")
    print(f"File: {filename}")
    print(f"Size: {len(chunk)} characters, {len(lines)} lines")
    print("Preview:")
    for line in preview_lines:
        print(f"  {line}")
    if len(lines) > 5:
        print(f"  ... ({len(lines) - 5} more lines)")
    print()


def write_chunk_file(
    chunk: str, filepath: Path, overwrite: bool, verbosity: int
) -> None:
    """Write a chunk to a file."""
    logger = logging.getLogger(__name__)

    if filepath.exists() and not overwrite:
        logger.warning(f"File exists, skipping: {filepath}")
        return

    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chunk)

        if verbosity >= 1:
            logger.info(f"Written: {filepath} ({len(chunk)} chars)")
        elif verbosity >= 2:
            lines = len(chunk.split("\n"))
            logger.info(f"Written: {filepath} ({len(chunk)} chars, {lines} lines)")

    except Exception as e:
        logger.error(f"Failed to write {filepath}: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Chunk EPUB and other files into semantically coherent segments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_base_args(parser)

    parser.add_argument(
        "-o",
        "--out",
        type=str,
        help="Output path pattern (default: {input_stem}_chunks/{input_stem}_{chunk_num}.{ext})",
    )

    parser.add_argument(
        "--pad",
        default="auto",
        help="Padding for chunk numbers: 'auto', 'none', or integer (default: auto)",
    )

    parser.add_argument(
        "--format",
        choices=["md", "txt", "html"],
        default="md",
        help="Output format; pandoc required for md/txt",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preview what would be created without writing files (default: %(default)s)",
    )

    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing files (default: %(default)s)",
    )

    parser.add_argument(
        "--file-boundary",
        choices=["none", "soft", "hard"],
        default="soft",
        help="How to handle boundaries between multiple input files: "
        "'none' (no boundaries), 'soft' (section breaks), "
        "'hard' (always end chunks at file boundaries) (default: %(default)s)",
    )

    return parser.parse_args()


def main() -> int:
    """Main CLI entry point."""
    try:
        args = parse_arguments()

        # Setup logging
        setup_logging(args.verbose)
        logger = logging.getLogger(__name__)

        # Validate input files
        for input_file in args.input_files:
            if not input_file.exists():
                logger.error(f"Input file does not exist: {input_file}")
                return 1
            if not input_file.is_file():
                logger.error(f"Path is not a file: {input_file}")
                return 1

        # Generate output path if not provided (use first input file's stem)
        first_input = args.input_files[0]
        if not args.out:
            input_stem = first_input.stem
            args.out = f"{input_stem}_chunks/{input_stem}_{{chunk_num}}.{{ext}}"
        else:
            # If path ends with directory separator, add input basename
            if args.out.endswith(("/", "\\")):
                input_stem = first_input.stem
                args.out = f"{args.out}{input_stem}_{{chunk_num}}.{{ext}}"
            else:
                # Ensure required placeholders are present
                if "{chunk_num}" not in args.out:
                    path = Path(args.out)
                    if path.suffix:
                        # Has extension, insert chunk_num before it
                        stem = path.stem
                        suffix = path.suffix
                        args.out = str(path.parent / f"{stem}_{{chunk_num}}{suffix}")
                    else:
                        # No extension, add chunk_num and ext
                        args.out = f"{args.out}_{{chunk_num}}.{{ext}}"

                if "{ext}" not in args.out:
                    # Add extension placeholder if missing
                    path = Path(args.out)
                    if not path.suffix:
                        args.out = f"{args.out}.{{ext}}"

        if args.verbose >= 1:
            logger.info(
                f"Processing {len(args.input_files)} file(s): {[str(f) for f in args.input_files]}"
            )
            logger.info(f"Output pattern: {args.out}")
            logger.info(f"File boundary mode: {args.file_boundary}")

        # Use centralized chunking for all file types
        logger.info("Chunking files...")
        file_boundary = FileBoundary(args.file_boundary)
        chunks = chunk_files(
            args.input_files,
            max_chunk_chars=args.max_chunk_chars,
            min_chunk_chars=args.min_chunk_chars,
            file_boundary=file_boundary,
            format=args.format,
            skip_index_p=args.skip_index,
        )

        if not chunks:
            logger.warning("No chunks generated from input file(s)")
            return 0

        # Calculate padding
        padding = calculate_padding(len(chunks), args.pad)

        if args.verbose >= 1:
            logger.info(f"Generated {len(chunks)} chunks")
            if padding:
                logger.info(f"Using {padding}-digit padding")
            else:
                logger.info("No padding for chunk numbers")

        # Process chunks
        input_stem = first_input.stem
        total_written = 0
        lens = [len(c) for c in chunks]
        total_chars = sum(lens)
        min_len = min(lens) if lens else 0
        max_len = max(lens) if lens else 0
        avg_len = (total_chars / len(lens)) if lens else 0

        for i, chunk in enumerate(chunks):
            # Generate filename
            filename_pattern = args.out.replace("{input_stem}", input_stem)
            if padding:
                chunk_num = str(i + 1).zfill(padding)
            else:
                chunk_num = str(i + 1)

            extension = get_file_extension(args.format)
            filename = filename_pattern.replace("{chunk_num}", chunk_num).replace(
                "{ext}", extension
            )
            filepath = Path(filename)

            if args.dry_run:
                print_chunk_preview(chunk, filepath, i)
            else:
                write_chunk_file(chunk, filepath, args.overwrite, args.verbose)
                total_written += 1

        # Summary
        if args.dry_run:
            print(
                f"Dry run completed. Would create {len(chunks)} files totaling {total_chars} characters (min={min_len}, avg={avg_len:.1f}, max={max_len})."
            )
        else:
            if args.verbose >= 1:
                logger.info(
                    f"Successfully written {total_written} chunk files (total characters: {total_chars}; min={min_len}, avg={avg_len:.1f}, max={max_len})"
                )

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        logging.getLogger(__name__).error(f"Error: {e}")
        if args.verbose >= 1:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
