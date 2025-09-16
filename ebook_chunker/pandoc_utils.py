import subprocess
import shutil
from typing import Optional


def _check_pandoc_available() -> bool:
    """Check if pandoc is available in the system."""
    return shutil.which("pandoc") is not None


def convert_text(
    text: str, *, from_format: str = "html", to_format: str = "markdown"
) -> str:
    """
    Convert text from one format to another using pandoc.

    Args:
        text: The input text to convert
        from_format: Source format (html, markdown, plain, etc.)
        to_format: Target format (markdown, plain, html, etc.)

    Returns:
        Converted text

    Raises:
        RuntimeError: If pandoc is not available or conversion fails
    """
    if not _check_pandoc_available():
        raise RuntimeError(
            "pandoc is not available. Please install pandoc to use format conversion."
        )

    if from_format == to_format:
        return text

    # Map common format names to pandoc format identifiers
    format_mapping = {
        "md": "markdown",
        "txt": "plain",
        "text": "plain",
        "plain": "plain",
        "markdown": "markdown",
        "html": "html",
    }

    pandoc_from = format_mapping.get(from_format, from_format)
    pandoc_to = format_mapping.get(to_format, to_format)

    try:
        result = subprocess.run(
            ["pandoc", "-f", pandoc_from, "-t", pandoc_to],
            input=text,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Pandoc conversion failed: {e.stderr.decode() if e.stderr else str(e)}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError("pandoc command not found. Please install pandoc.") from None


def get_file_extension(format_name: str) -> str:
    """Get appropriate file extension for a format."""
    extensions = {
        "markdown": "md",
        "md": "md",
        "plain": "txt",
        "txt": "txt",
        "text": "txt",
        "html": "html",
    }
    return extensions.get(format_name, format_name)
