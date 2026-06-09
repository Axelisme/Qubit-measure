"""Sensible default starting directories for the file dialogs.

Derived from the project (chip / qubit / result_dir / database_path) so the user
doesn't have to navigate from scratch. The helper returns the nearest existing
ancestor of the intended directory, so a dialog opens close to the right place
even before any file exists there.
"""

from __future__ import annotations

import os

from zcu_tools.gui.project import ProjectInfo


def _nearest_existing(path: str) -> str:
    """The deepest existing ancestor of ``path`` (``path`` itself if it exists).

    Returns "" if nothing in the chain exists (then Qt opens the cwd).
    """
    path = os.path.abspath(path) if path else ""
    while path and not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:  # reached the filesystem root
            return ""
        path = parent
    return path


def raw_onetone_dir(project: ProjectInfo) -> str:
    """Where raw one-tone hdf5 files live — the project's database_path root."""
    return _nearest_existing(project.database_path)


def params_dir(project: ProjectInfo) -> str:
    """Where params.json is read/written — the project's result_dir."""
    return _nearest_existing(project.result_dir)
