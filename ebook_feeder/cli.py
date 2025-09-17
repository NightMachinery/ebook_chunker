import argparse
import logging
import os
import sys
from pathlib import Path

from ebook_chunker.base_cli import add_base_args, setup_logging
from ebook_chunker.epub_chunker import chunk_epub
from .constants import (
    PROMPT_CURRENT_INPUT_PLACEHOLDER,
    PROMPT_CURRENT_OUTPUT_PLACEHOLDER,
)
from .models import ChunkResult


def _render_prompt(template: str, *, current_output: str, current_input: str) -> str:
    prompt = template.replace(PROMPT_CURRENT_OUTPUT_PLACEHOLDER, current_output)
    prompt = prompt.replace(PROMPT_CURRENT_INPUT_PLACEHOLDER, current_input)
    return prompt


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Feed EPUB chunks to an LLM (Gemini Flash) and accumulate outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_base_args(parser)

    parser.add_argument(
        "--prompt",
        type=Path,
        required=True,
        help=(
            f"Path to the prompt template file. Must contain "
            f"{PROMPT_CURRENT_INPUT_PLACEHOLDER} and {PROMPT_CURRENT_OUTPUT_PLACEHOLDER}."
        ),
    )

    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Output markdown file (default: {input_stem}_fed.md)",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preview what would be done without writing files or calling APIs (default: %(default)s)",
    )

    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing output files (default: %(default)s)",
    )

    return parser.parse_args()


def main() -> int:
    args = None
    try:
        args = parse_arguments()
        setup_logging(args.verbose)
        logger = logging.getLogger(__name__)

        if not args.epub_path.exists() or not args.epub_path.is_file():
            logger.error(f"EPUB file not found: {args.epub_path}")
            return 1

        if not args.prompt.exists() or not args.prompt.is_file():
            logger.error(f"Prompt file not found: {args.prompt}")
            return 1

        prompt_template = args.prompt.read_text(encoding="utf-8")
        if PROMPT_CURRENT_INPUT_PLACEHOLDER not in prompt_template:
            logger.warning(
                f"Prompt missing placeholder: {PROMPT_CURRENT_INPUT_PLACEHOLDER}"
            )
        if PROMPT_CURRENT_OUTPUT_PLACEHOLDER not in prompt_template:
            logger.warning(
                f"Prompt missing placeholder: {PROMPT_CURRENT_OUTPUT_PLACEHOLDER}"
            )

        # Resolve output path
        if not args.out:
            args.out = args.epub_path.with_name(f"{args.epub_path.stem}_fed.md")
        elif args.out.is_dir():
            args.out = args.out / f"{args.epub_path.stem}_fed.md"

        if args.out.exists() and not args.overwrite:
            logger.error(f"Output exists: {args.out}. Use --overwrite to replace it.")
            return 1

        logger.info(f"Processing: {args.epub_path}")
        logger.info(f"Using prompt: {args.prompt}")
        logger.info(f"Output: {args.out}")

        logger.info("Chunking EPUB file...")
        chunks = chunk_epub(
            str(args.epub_path),
            max_chunk_chars=args.max_chunk_chars,
            min_chunk_chars=args.min_chunk_chars,
            format="md",
            skip_index_p=args.skip_index,
        )

        if not chunks:
            logger.warning("No chunks generated.")
            return 0

        accumulated_output = ""

        if not args.dry_run:
            try:
                import litellm  # type: ignore
            except Exception as e:
                logger.error(
                    "litellm is required to call the LLM. Please install it (e.g. poetry add litellm)."
                )
                logger.debug(f"Import error: {e}")
                return 1

            # Allow model override via env var for flexibility
            model_name = os.environ.get("EBOOK_FEEDER_MODEL", "gemini/gemini-2.5-flash")

        for i, chunk in enumerate(chunks, start=1):
            logger.info(f"Processing chunk {i}/{len(chunks)}...")
            prompt = _render_prompt(
                prompt_template,
                current_output=accumulated_output,
                current_input=chunk,
            )

            if args.dry_run:
                print(f"\n--- Chunk {i}/{len(chunks)} ---")
                print("PROMPT:")
                print(prompt)
                print("---")
                if accumulated_output:
                    accumulated_output += "\n\n"
                accumulated_output += (
                    f"[dry-run structured output for chunk {i} (content)]"
                )
                continue

            try:
                # Always structured output via Pydantic schema
                import json as _json

                sys_msg = (
                    "You are a precise assistant. Respond only for CURRENT_INPUT; do not restate CURRENT_OUTPUT. "
                    "Return a strict JSON object that matches the provided schema. No extra keys or text."
                )
                schema = ChunkResult.model_json_schema()
                resp = litellm.completion(
                    model=model_name,
                    temperature=0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "ChunkResult", "schema": schema},
                    },
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = resp.choices[0].message.content  # type: ignore[attr-defined]
                text = content if isinstance(content, str) else str(content)
                t = text.strip()
                if t.startswith("```"):
                    lines = t.splitlines()
                    if lines and lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    t = "\n".join(lines).strip()
                obj = _json.loads(t)
                model_obj = ChunkResult.model_validate(obj)
                new_piece = model_obj.content
            except Exception as e:
                logger.error(f"LLM call failed on chunk {i}: {e}")
                return 1

            if accumulated_output:
                accumulated_output += "\n\n" + new_piece
            else:
                accumulated_output = new_piece

        if args.dry_run:
            logger.info("Dry run finished. No files were written.")
            return 0

        logger.info(f"Writing final output to {args.out}...")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(accumulated_output, encoding="utf-8")
        logger.info("Done.")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        logging.getLogger(__name__).error(f"Error: {e}")
        if args and getattr(args, "verbose", 0) >= 3:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
