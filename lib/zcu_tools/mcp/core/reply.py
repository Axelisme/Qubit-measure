"""Per-call MCP results with inline PNG content, independent of file lifetimes."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PngImage:
    """PNG bytes owned by this reply; no file access or cleanup responsibility."""

    data: bytes


@dataclass(frozen=True)
class ToolReply:
    """JSON-compatible structured data followed by this call's ordered images."""

    data: dict[str, Any]
    images: tuple[PngImage, ...] = ()
