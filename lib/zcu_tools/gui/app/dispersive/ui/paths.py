"""Starting directory for raw one-tone browsing; no entry layout is inferred."""

from pathlib import Path

from zcu_tools.gui.widgets.file_dialog import nearest_existing


def raw_onetone_dir(project_root: str) -> str:
    """Start browsing at project_root/Database, or its nearest existing ancestor."""
    return nearest_existing(str(Path(project_root) / "Database"))
