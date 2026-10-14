"""JSON guide contract shared by the GUI producer and MCP consumer."""

from typing import TypedDict


class AdapterGuideView(TypedDict):
    """Complete guide text and the directory for its relative image references."""

    markdown: str
    """Unparsed Markdown source, or an unwritten-guide notice."""
    resource_dir: str | None
    """Absolute GUI-host directory path, or null when no guide is declared."""
