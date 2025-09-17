from pydantic import BaseModel, Field


class ChunkResult(BaseModel):
    content: str = Field(
        description=(
            "Primary output for CURRENT_INPUT only, no repetition of accumulated output, "
            "no extra commentary or pre/post text."
        )
    )
