from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkResult(BaseModel):
    # Primary output the app will append to the accumulated output
    content: str = Field(
        description=(
            "Primary output for CURRENT_INPUT only; do not repeat accumulated output; "
            "avoid meta-commentary."
        )
    )

    # Optional fields to capture anything else the model might want to say
    notes: Optional[str] = Field(
        default=None,
        description=("Optional side notes or comments that should NOT be in content."),
    )
    warnings: Optional[str] = Field(
        default=None, description="Optional cautions or boundaries; not for content."
    )
    citations: Optional[List[str]] = Field(
        default=None, description="Optional references or citations; not for content."
    )
    tags: Optional[List[str]] = Field(
        default=None, description="Optional tags/keywords; not for content."
    )
    title: Optional[str] = Field(
        default=None,
        description="Optional short title for this chunk; not for content.",
    )
    meta: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional structured metadata; not for content."
    )
