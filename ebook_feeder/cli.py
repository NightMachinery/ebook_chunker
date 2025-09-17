import litellm

litellm.turn_off_message_logging = True
litellm.suppress_debug_info = True
litellm.set_verbose = False

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Deque, Optional

from ebook_chunker.base_cli import add_base_args, setup_logging
from ebook_chunker.epub_chunker import chunk_epub
from .constants import (
    PROMPT_CURRENT_INPUT_PLACEHOLDER,
    PROMPT_CURRENT_OUTPUT_PLACEHOLDER,
    MAX_RETRIES_DEFAULT,
    RETRY_DELAY_DEFAULT,
)
from .models import ChunkResult


def _render_prompt(template: str, *, current_output: str, current_input: str) -> str:
    prompt = template.replace(PROMPT_CURRENT_OUTPUT_PLACEHOLDER, current_output)
    prompt = prompt.replace(PROMPT_CURRENT_INPUT_PLACEHOLDER, current_input)
    return prompt


def _load_api_key_queue(*, file_path: Path) -> Deque[str]:
    lines = file_path.read_text(encoding="utf-8").splitlines()
    keys = []
    for raw in lines:
        candidate = raw.strip()
        if not candidate or candidate.startswith("#"):
            continue
        keys.append(candidate)
    if not keys:
        raise ValueError("Gemini API key file contains no usable keys")
    return deque(keys)


class ClipboardUnavailableError(RuntimeError):
    pass


@dataclass
class ClipboardAdapter:
    copy: Callable[[str], None]
    paste: Callable[[], str]


def _describe_run_mode(*, interactive: bool) -> str:
    return "interactive" if interactive else "automatic"


def _build_clipboard_adapter(
    *,
    copy_fn: Optional[Callable[[str], None]] = None,
    paste_fn: Optional[Callable[[], str]] = None,
) -> ClipboardAdapter:
    if (copy_fn is None) ^ (paste_fn is None):
        raise ValueError(
            "copy_fn and paste_fn must both be provided or omitted together"
        )

    if copy_fn and paste_fn:
        return ClipboardAdapter(copy=copy_fn, paste=paste_fn)

    try:  # Prefer pyperclip if available
        import pyperclip  # type: ignore

        return ClipboardAdapter(copy=pyperclip.copy, paste=pyperclip.paste)
    except Exception:  # noqa: BLE001 - fallback to platform tools
        pass

    platform = sys.platform
    if platform == "darwin" and shutil.which("pbcopy") and shutil.which("pbpaste"):

        def _copy_darwin(text: str) -> None:
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)

        def _paste_darwin() -> str:
            result = subprocess.run(["pbpaste"], capture_output=True, check=True)
            return result.stdout.decode("utf-8")

        return ClipboardAdapter(copy=_copy_darwin, paste=_paste_darwin)

    if platform.startswith("linux"):
        if shutil.which("wl-copy") and shutil.which("wl-paste"):

            def _copy_wl(text: str) -> None:
                subprocess.run(["wl-copy"], input=text.encode("utf-8"), check=True)

            def _paste_wl() -> str:
                result = subprocess.run(["wl-paste"], capture_output=True, check=True)
                return result.stdout.decode("utf-8")

            return ClipboardAdapter(copy=_copy_wl, paste=_paste_wl)

        if shutil.which("xclip"):

            def _copy_xclip(text: str) -> None:
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=text.encode("utf-8"),
                    check=True,
                )

            def _paste_xclip() -> str:
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True,
                    check=True,
                )
                return result.stdout.decode("utf-8")

            return ClipboardAdapter(copy=_copy_xclip, paste=_paste_xclip)

        if shutil.which("xsel"):

            def _copy_xsel(text: str) -> None:
                subprocess.run(
                    ["xsel", "--clipboard", "--input"],
                    input=text.encode("utf-8"),
                    check=True,
                )

            def _paste_xsel() -> str:
                result = subprocess.run(
                    ["xsel", "--clipboard", "--output"],
                    capture_output=True,
                    check=True,
                )
                return result.stdout.decode("utf-8")

            return ClipboardAdapter(copy=_copy_xsel, paste=_paste_xsel)

    if platform.startswith("win"):
        if shutil.which("clip"):

            def _copy_windows(text: str) -> None:
                subprocess.run(
                    ["clip"],
                    input=text.encode("utf-16-le"),
                    check=True,
                )

            def _paste_windows() -> str:
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Get-Clipboard",
                    ],
                    capture_output=True,
                    check=True,
                )
                return result.stdout.decode("utf-8")

            return ClipboardAdapter(copy=_copy_windows, paste=_paste_windows)

    raise ClipboardUnavailableError(
        "No clipboard integration available. Install 'pyperclip' or ensure a system clipboard utility is present."
    )


def _collect_interactive_response(
    *,
    chunk_index: int,
    total_chunks: int,
    prompt: str,
    clipboard: ClipboardAdapter,
) -> str:
    def _copy_prompt() -> None:
        clipboard.copy(prompt)
        print(
            f"[chunk {chunk_index}/{total_chunks}] Prompt copied to clipboard. Copy the LLM response and press Enter to capture it."
        )

    _copy_prompt()

    while True:
        user_entry = (
            input("Press Enter to read clipboard, or type 'y' to recopy prompt: ")
            .strip()
            .lower()
        )
        if user_entry == "y":
            _copy_prompt()
            continue

        if user_entry not in {"", "enter"}:  # 'enter' supports accidental typing
            print(
                "Unrecognized input. Press Enter to read clipboard or 'y' to recopy prompt."
            )
            continue

        answer = clipboard.paste()
        if not answer.strip():
            print("Clipboard is empty. Copy the LLM reply and press Enter again.")
            continue

        if answer.strip() == prompt.strip():
            print(
                "Clipboard still contains the prompt. Copy the LLM reply before pressing Enter."
            )
            continue

        preview_lines = 3
        lines = answer.splitlines()
        head = lines[:preview_lines]
        tail = lines[-preview_lines:] if len(lines) > preview_lines else []
        print("Captured response preview:")
        if head:
            print(">>> first lines >>>")
            print("\n".join(head))
        else:
            print("(response begins with an empty line)")
        if tail:
            print("... (middle omitted) ...")
            print("<<< last lines <<<")
            print("\n".join(tail))
        elif not head:
            print("(response is empty)")

        return answer


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
        "-m",
        "--model",
        type=str,
        help=(
            "Model name to use (default: env EBOOK_FEEDER_MODEL or 'gemini/gemini-2.5-flash')"
        ),
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the model (default: %(default)s)",
    )

    parser.add_argument(
        "--structured-outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request JSON-structured outputs (default: %(default)s)",
    )

    parser.add_argument(
        "--output-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend an HTML comment with chunk metadata before each section (default: %(default)s)",
    )

    parser.add_argument(
        "-i",
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable manual interactive mode driven via the clipboard (default: %(default)s)",
    )

    parser.add_argument(
        "--temp-dir",
        type=Path,
        help=(
            "Directory to save intermediate model responses. If not provided, a new "
            "temporary directory is created under the OS temp directory."
        ),
    )

    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Resume from a previous temp directory containing saved chunk responses. "
            "Must match the same input and chunking configuration."
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES_DEFAULT,
        help="Max retries for model calls (default: %(default)s)",
    )

    parser.add_argument(
        "--retry-delay",
        type=float,
        default=RETRY_DELAY_DEFAULT,
        help="Initial backoff delay in seconds (default: %(default)s)",
    )

    parser.add_argument(
        "--gemini-api-key-file",
        type=Path,
        help="Path to a file containing Gemini API keys, one per line",
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

        model_name = args.model or os.environ.get("EBOOK_FEEDER_MODEL")
        if not model_name:
            model_name = (
                "interactive/manual" if args.interactive else "gemini/gemini-2.5-flash"
            )
        run_mode = _describe_run_mode(interactive=args.interactive)

        # Resolve output path
        if not args.out:
            args.out = args.epub_path.with_name(f"{args.epub_path.stem}_fed.md")
        elif args.out.is_dir():
            args.out = args.out / f"{args.epub_path.stem}_fed.md"

        if args.out.exists() and not args.overwrite:
            logger.error(f"Output exists: {args.out}. Use --overwrite to replace it.")
            return 1

        api_key_queue: Optional[Deque[str]] = None
        if args.gemini_api_key_file:
            key_file = args.gemini_api_key_file.expanduser()
            if not key_file.exists() or not key_file.is_file():
                logger.error(f"Gemini API key file not found or not a file: {key_file}")
                return 1
            try:
                api_key_queue = _load_api_key_queue(file_path=key_file)
            except ValueError as exc:
                logger.error(str(exc))
                return 1
            logger.info(
                "Loaded %d Gemini API key(s) from %s",
                len(api_key_queue),
                key_file,
            )

        clipboard_adapter: Optional[ClipboardAdapter] = None
        if args.interactive:
            try:
                clipboard_adapter = _build_clipboard_adapter()
            except ClipboardUnavailableError as exc:
                logger.error(f"Interactive mode requires clipboard support: {exc}")
                return 1
            logger.info(
                "Interactive mode enabled; copy/paste will use the system clipboard."
            )

        # Resolve temp directory and optionally resume
        temp_dir: Path
        if args.resume_from:
            temp_dir = Path(args.resume_from)
            if not temp_dir.exists() or not temp_dir.is_dir():
                logger.error(
                    f"--resume-from path not found or not a directory: {temp_dir}"
                )
                return 1
        else:
            if args.temp_dir:
                temp_dir = Path(args.temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)
            else:
                base = Path(tempfile.gettempdir())
                temp_dir = (
                    base
                    / f"ebook_feeder_{int(time.time())}_{random.randint(1000,9999)}"
                )
                temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Temp dir: {temp_dir}")
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
        accumulated_prompt_output = ""

        # Save run metadata and prompt template to temp dir (non-dry-run)
        if not args.dry_run:
            try:
                prompt_copy = temp_dir / "prompt.md"
                if not prompt_copy.exists():
                    prompt_copy.write_text(prompt_template, encoding="utf-8")

                run_meta_path = temp_dir / "run.json"
                selected_model = model_name
                existing_meta: dict[str, Any] = {}
                if run_meta_path.exists():
                    try:
                        existing_meta = json.loads(
                            run_meta_path.read_text(encoding="utf-8")
                        )
                    except Exception:
                        existing_meta = {}

                created_ts = existing_meta.get("created")
                if not isinstance(created_ts, str):
                    created_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

                run_meta = {
                    "created": created_ts,
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "mode": run_mode,
                    "epub_path": str(args.epub_path),
                    "input_stem": args.epub_path.stem,
                    "max_chunk_chars": int(args.max_chunk_chars),
                    "min_chunk_chars": int(args.min_chunk_chars),
                    "skip_index": bool(args.skip_index),
                    "structured_outputs": bool(args.structured_outputs),
                    "model": selected_model,
                    "temperature": float(args.temperature),
                    "prompt_path": str(args.prompt),
                    "prompt_sha256": hashlib.sha256(
                        prompt_template.encode("utf-8")
                    ).hexdigest(),
                }
                run_meta_path.write_text(
                    json.dumps(run_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        if not args.dry_run and not args.interactive:
            try:
                import litellm  # type: ignore
            except Exception as e:
                logger.error(
                    "litellm is required to call the LLM. Please install it (e.g. poetry add litellm)."
                )
                logger.debug(f"Import error: {e}")
                return 1

        # Helper to hash a chunk for resume integrity checks
        def _chunk_hash(text: str) -> str:
            return hashlib.sha256(text.encode("utf-8")).hexdigest()

        def _build_metadata_comment(
            *,
            chunk_index: int,
            model_name: str,
            structured: bool,
            chunk_hash: str,
            timestamp: str,
            usage: Optional[dict],
        ) -> str:
            meta_parts = [
                f"chunk:{chunk_index:04d}",
                f"model:{model_name}",
                f"structured:{structured}",
                f"hash:{chunk_hash}",
                f"ts:{timestamp}",
            ]
            if usage and (
                usage.get("prompt_tokens") is not None
                or usage.get("completion_tokens") is not None
            ):
                meta_parts.append(
                    "tokens:"
                    + ",".join(
                        [
                            f"{usage.get('prompt_tokens') or '-'}p",
                            f"{usage.get('completion_tokens') or '-'}c",
                            f"{usage.get('total_tokens') or '-'}t",
                        ]
                    )
                )
            return "\n<!-- " + " | ".join(meta_parts) + " -->\n\n"

        # If resuming, pre-load saved results and compute start index
        start_index = 1
        if args.resume_from:
            # Validate run metadata compatibility if present
            try:
                meta_path = temp_dir / "run.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    mismatches = []
                    saved_mode = meta.get("mode")
                    if isinstance(saved_mode, str):
                        if saved_mode not in {"interactive", "automatic"}:
                            logger.warning(
                                "Unknown saved run mode '%s' in %s; using current mode %s.",
                                saved_mode,
                                meta_path,
                                run_mode,
                            )
                        elif saved_mode != run_mode:
                            logger.info(
                                "Resume mode differs (saved %s, current %s); continuing with current mode.",
                                saved_mode,
                                run_mode,
                            )
                    if meta.get("input_stem") != args.epub_path.stem:
                        mismatches.append("input_stem")
                    if int(meta.get("max_chunk_chars", -1)) != int(
                        args.max_chunk_chars
                    ):
                        mismatches.append("max_chunk_chars")
                    if int(meta.get("min_chunk_chars", -1)) != int(
                        args.min_chunk_chars
                    ):
                        mismatches.append("min_chunk_chars")
                    if bool(meta.get("skip_index")) != bool(args.skip_index):
                        mismatches.append("skip_index")
                    if bool(meta.get("structured_outputs")) != bool(
                        args.structured_outputs
                    ):
                        mismatches.append("structured_outputs")
                    # Compare prompt hash
                    expected_prompt_sha = meta.get("prompt_sha256")
                    current_prompt_sha = hashlib.sha256(
                        prompt_template.encode("utf-8")
                    ).hexdigest()
                    if (
                        expected_prompt_sha
                        and expected_prompt_sha != current_prompt_sha
                    ):
                        mismatches.append("prompt_sha256")
                    if mismatches:
                        logger.error(
                            "Resume run metadata mismatch in: " + ", ".join(mismatches)
                        )
                        return 1
            except Exception as e:
                logger.warning(f"Failed to load/validate run metadata: {e}")
            saved_files = sorted(temp_dir.glob("chunk_*.json"))
            contiguous = 0
            for fpath in saved_files:
                try:
                    with fpath.open("r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    idx = int(data.get("index"))
                    if idx != contiguous + 1:
                        break
                    expected_hash = data.get("chunk_hash")
                    current_hash = _chunk_hash(chunks[idx - 1])
                    if expected_hash != current_hash:
                        logger.error(
                            f"Resume mismatch at chunk {idx}: saved hash differs from current run."
                        )
                        return 1
                    model_obj = ChunkResult.model_validate(data.get("result", {}))
                    content_piece = model_obj.content
                    if accumulated_prompt_output:
                        accumulated_prompt_output += "\n\n" + content_piece
                    else:
                        accumulated_prompt_output = content_piece

                    section_text_saved = content_piece
                    if args.output_metadata:
                        ts_saved = data.get("ts")
                        if not isinstance(ts_saved, str):
                            ts_saved = datetime.now(timezone.utc).isoformat()
                        usage_saved = data.get("usage")
                        usage_meta = (
                            usage_saved if isinstance(usage_saved, dict) else None
                        )
                        meta_header = _build_metadata_comment(
                            chunk_index=idx,
                            model_name=str(
                                data.get("model")
                                or args.model
                                or os.environ.get("EBOOK_FEEDER_MODEL", "unknown")
                            ),
                            structured=bool(data.get("structured")),
                            chunk_hash=current_hash,
                            timestamp=ts_saved,
                            usage=usage_meta,
                        )
                        section_text_saved = meta_header + content_piece
                    if accumulated_output:
                        accumulated_output += "\n\n" + section_text_saved
                    else:
                        accumulated_output = section_text_saved
                    contiguous += 1
                except Exception as e:
                    logger.warning(f"Skipping unreadable saved file {fpath.name}: {e}")
                    break
            if contiguous > 0:
                logger.info(
                    f"Resuming from chunk {contiguous + 1} (loaded {contiguous})."
                )
                start_index = contiguous + 1

        def _save_intermediate(
            *,
            directory: Path,
            index: int,
            chunk_hash: str,
            raw_text: str,
            result: ChunkResult,
            model_name: str,
            structured: bool,
            usage: Optional[dict],
        ) -> None:
            ts = datetime.now(timezone.utc).isoformat()
            out = {
                "index": index,
                "chunk_hash": chunk_hash,
                "model": model_name,
                "result": result.model_dump(),
                "raw": raw_text,
                "structured": structured,
                "usage": usage,
                "ts": ts,
            }
            fp = directory / f"chunk_{index:04d}.json"
            fp.write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        def _call_with_retries(
            fn: Callable[[Optional[str]], Any],
            *,
            max_retries: int,
            delay: float,
            key_queue: Optional[Deque[str]] = None,
        ) -> Any:
            attempts = max(0, int(max_retries)) + 1
            cur = max(0.0, float(delay))
            for i in range(attempts):
                api_key = None
                if key_queue:
                    api_key = key_queue[0]
                    os.environ["GEMINI_API_KEY"] = api_key
                try:
                    return fn(api_key=api_key)
                except Exception as e:  # noqa: PERF203
                    if key_queue:
                        key_queue.append(key_queue.popleft())
                        logger.debug(
                            "Rotated Gemini API key after failure on attempt %d.",
                            i + 1,
                        )
                    if i >= attempts - 1:
                        raise
                    jitter = random.uniform(0, cur * 0.25)
                    wait_s = cur + jitter
                    logger.warning(
                        f"Model call failed (attempt {i+1}/{attempts}), retrying in {wait_s:.2f}s: {e}"
                    )
                    time.sleep(wait_s)
                    cur *= 2.0

        # Ensure chunk input directory exists (non-dry-run)
        chunks_dir = temp_dir / "chunks"
        if not args.dry_run:
            chunks_dir.mkdir(parents=True, exist_ok=True)

        for i, chunk in enumerate(chunks, start=1):
            if i < start_index:
                continue
            logger.info(f"Processing chunk {i}/{len(chunks)}...")
            prompt = _render_prompt(
                prompt_template,
                current_output=accumulated_prompt_output,
                current_input=chunk,
            )

            if args.dry_run:
                print(f"\n--- Chunk {i}/{len(chunks)} ---")
                print("PROMPT:")
                print(prompt)
                print("---")
                placeholder = f"[dry-run structured output for chunk {i} (content)]"
                if accumulated_prompt_output:
                    accumulated_prompt_output += "\n\n"
                accumulated_prompt_output += placeholder
                if accumulated_output:
                    accumulated_output += "\n\n"
                accumulated_output += placeholder
                continue

            usage_rec: Optional[dict] = None

            try:
                if args.interactive:
                    if clipboard_adapter is None:
                        raise ClipboardUnavailableError(
                            "Interactive mode requires an initialized clipboard adapter."
                        )
                    raw_text = _collect_interactive_response(
                        chunk_index=i,
                        total_chunks=len(chunks),
                        prompt=prompt,
                        clipboard=clipboard_adapter,
                    )
                else:
                    # Call model (structured or freeform) with retries
                    def _do_call(*, api_key: Optional[str]) -> Any:
                        sys_msg_structured = (
                            "You are a precise assistant. Respond only for CURRENT_INPUT; do not restate CURRENT_OUTPUT. "
                            "Return a strict JSON object that matches the provided schema. No extra keys or text."
                        )
                        sys_msg_freeform = "You are a precise assistant. Respond only for CURRENT_INPUT; do not restate CURRENT_OUTPUT."
                        kwargs = {
                            "model": model_name,
                            "temperature": args.temperature,
                        }
                        if api_key:
                            kwargs["api_key"] = api_key
                        if args.structured_outputs:
                            schema = ChunkResult.model_json_schema()
                            kwargs["response_format"] = {
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "ChunkResult",
                                    "schema": schema,
                                },
                            }
                            kwargs["messages"] = [
                                {"role": "system", "content": sys_msg_structured},
                                {"role": "user", "content": prompt},
                            ]
                        else:
                            kwargs["messages"] = [
                                {"role": "system", "content": sys_msg_freeform},
                                {"role": "user", "content": prompt},
                            ]
                        return litellm.completion(**kwargs)

                    resp = _call_with_retries(
                        _do_call,
                        max_retries=args.max_retries,
                        delay=args.retry_delay,
                        key_queue=api_key_queue,
                    )

                    # Extract content string from response
                    content = resp.choices[0].message.content  # type: ignore[attr-defined]
                    raw_text = content if isinstance(content, str) else str(content)

                    # Optional usage summary
                    usage = getattr(resp, "usage", None)
                    if usage is not None:
                        try:
                            usage_rec = {
                                "prompt_tokens": (
                                    getattr(usage, "prompt_tokens", None)
                                    if not isinstance(usage, dict)
                                    else usage.get("prompt_tokens")
                                ),
                                "completion_tokens": (
                                    getattr(usage, "completion_tokens", None)
                                    if not isinstance(usage, dict)
                                    else usage.get("completion_tokens")
                                ),
                                "total_tokens": (
                                    getattr(usage, "total_tokens", None)
                                    if not isinstance(usage, dict)
                                    else usage.get("total_tokens")
                                ),
                            }
                        except Exception:
                            usage_rec = None

                if usage_rec is not None:
                    prompt_tokens = usage_rec.get("prompt_tokens")
                    completion_tokens = usage_rec.get("completion_tokens")
                    total_tokens = usage_rec.get("total_tokens")
                    logger.info(
                        "Chunk %04d token usage: prompt=%s completion=%s total=%s",
                        i,
                        prompt_tokens if prompt_tokens is not None else "-",
                        completion_tokens if completion_tokens is not None else "-",
                        total_tokens if total_tokens is not None else "-",
                    )

                if args.structured_outputs:
                    # Parse JSON and validate via Pydantic
                    t = raw_text.strip()
                    if t.startswith("```"):
                        lines = t.splitlines()
                        if lines and lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        t = "\n".join(lines).strip()
                    obj = json.loads(t)
                    model_obj = ChunkResult.model_validate(obj)
                    new_piece = model_obj.content
                else:
                    model_obj = ChunkResult(content=raw_text)
                    new_piece = raw_text

                chunk_hash = _chunk_hash(chunk)
                ts = datetime.now(timezone.utc).isoformat()
                section_text_for_prompt = new_piece
                section_text_for_save = new_piece
                if args.output_metadata:
                    section_header = _build_metadata_comment(
                        chunk_index=i,
                        model_name=model_name,
                        structured=bool(args.structured_outputs),
                        chunk_hash=chunk_hash,
                        timestamp=ts,
                        usage=usage_rec,
                    )
                    section_text_for_save = section_header + new_piece

                # Save artifacts (chunk input, chunk prompt, intermediate JSON, accumulated)
                if not args.dry_run:
                    # Save input chunk
                    (chunks_dir / f"chunk_{i:04d}.md").write_text(
                        chunk, encoding="utf-8"
                    )
                    # Save rendered prompt for this chunk
                    (temp_dir / f"chunk_{i:04d}_prompt.md").write_text(
                        prompt, encoding="utf-8"
                    )

                    # Save intermediate JSON
                    _save_intermediate(
                        directory=temp_dir,
                        index=i,
                        chunk_hash=chunk_hash,
                        raw_text=raw_text,
                        result=model_obj,
                        model_name=model_name,
                        structured=bool(args.structured_outputs),
                        usage=usage_rec,
                    )
            except KeyboardInterrupt:
                logger.info("Interactive session interrupted by user.")
                return 130
            except Exception as e:
                if args.interactive:
                    logger.error(f"Interactive capture failed on chunk {i}: {e}")
                else:
                    logger.error(f"LLM call failed on chunk {i}: {e}")
                return 1

            if accumulated_prompt_output:
                accumulated_prompt_output += "\n\n" + section_text_for_prompt
            else:
                accumulated_prompt_output = section_text_for_prompt

            if accumulated_output:
                accumulated_output += "\n\n" + section_text_for_save
            else:
                accumulated_output = section_text_for_save

            # Save running accumulated output snapshot
            if not args.dry_run:
                (temp_dir / "accumulated.md").write_text(
                    accumulated_output, encoding="utf-8"
                )

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
        if args and getattr(args, "verbose", 0) >= 1:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
