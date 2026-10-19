"""Sensible default starting directories for the file dialogs.

Derived from the project (chip / qubit / result_dir / data_root) so the user
doesn't have to navigate from scratch. The helper returns the nearest existing
ancestor of the intended directory, so a dialog opens close to the right place
even before any file exists there.
"""

from __future__ import annotations

from zcu_tools.gui.app.analysis_project.project import ProjectInfo
from zcu_tools.gui.widgets.file_dialog import nearest_existing


def raw_onetone_dir(project: ProjectInfo) -> str:
    """Where raw one-tone hdf5 files live — the project's data_root root."""
    return nearest_existing(project.data_root)


def params_dir(project: ProjectInfo) -> str:
    """Where params.json is read/written — the project's result_dir."""
    return nearest_existing(project.result_dir)
