"""File-dialog starting directories for entry-scoped analysis."""

from __future__ import annotations

import os

from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget
from zcu_tools.gui.widgets.file_dialog import nearest_existing

_SIM_DB_REL = os.path.join("Database", "simulation")


def _sim_db_dir(root: str = "") -> str:
    return os.path.join(root, _SIM_DB_REL) if root else _SIM_DB_REL


def raw_spectrum_dir(project: AnalysisTarget | None) -> str:
    """Start at the selected entry's Database output root, or the current directory."""
    return nearest_existing(
        str(project.resource.database_path) if project is not None else "."
    )


def processed_spectrum_dir(project: AnalysisTarget | None) -> str:
    """Start at the selected entry's fluxdep intermediate directory."""
    target = project.resource.result_path / "fluxdep" if project is not None else "."
    return nearest_existing(str(target))


def database_dir(project: AnalysisTarget | None, root: str = "") -> str:
    """Start at bundled simulation databases, independent of the selected component."""
    del project
    return nearest_existing(_sim_db_dir(root))


def default_database_file(project: AnalysisTarget | None, root: str = "") -> str:
    """Return the first bundled fluxonium HDF5 database, or empty when none exists."""
    del project
    sim_dir = _sim_db_dir(root)
    if os.path.isdir(sim_dir):
        candidates = sorted(
            f
            for f in os.listdir(sim_dir)
            if f.startswith("fluxonium") and f.endswith((".h5", ".hdf5"))
        )
        if candidates:
            return os.path.join(sim_dir, candidates[0])
    return ""
