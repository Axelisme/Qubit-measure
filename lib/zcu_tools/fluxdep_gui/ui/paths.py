"""Sensible default starting directories for the file dialogs.

Derived from the project (chip / qubit / result_dir / database_path) so the user
doesn't have to navigate from scratch each time. Each helper returns a directory
that exists where possible (falling back up the tree, then to ""), so passing it
to a Qt file dialog just opens it at a useful place.
"""

from __future__ import annotations

import os

from zcu_tools.fluxdep_gui.state import ProjectInfo

# The repo's bundled simulation databases (relative to the cwd a launch uses).
_SIM_DB_DIR = os.path.join("Database", "simulation")


def _first_existing(*candidates: str) -> str:
    """The first candidate path that exists, else ""."""
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return ""


def raw_spectrum_dir(project: ProjectInfo) -> str:
    """Where raw spectrum hdf5 files live — the project's database_path root."""
    return _first_existing(project.database_path)


def processed_spectrum_dir(project: ProjectInfo) -> str:
    """Where exported spectrums.hdf5 lives — ``<result_dir>/data/fluxdep``."""
    if project.result_dir:
        return _first_existing(
            os.path.join(project.result_dir, "data", "fluxdep"),
            project.result_dir,
        )
    return ""


def database_dir(project: ProjectInfo) -> str:
    """Where the precomputed search database lives.

    Prefers the project's database_path, then the repo's bundled
    ``Database/simulation`` directory.
    """
    return _first_existing(project.database_path, _SIM_DB_DIR)


def params_dir(project: ProjectInfo) -> str:
    """Where params.json is written — the project's result_dir."""
    return _first_existing(project.result_dir)


def default_database_file(project: ProjectInfo) -> str:
    """A pre-fillable default database FILE, so the user often need not browse.

    If ``database_path`` points at a file, use it; if it's a directory (or the
    bundled ``Database/simulation`` exists), pick the first ``fluxonium*.h5`` in
    it. Returns "" when nothing suitable is found.
    """
    if project.database_path and os.path.isfile(project.database_path):
        return project.database_path
    for directory in (project.database_path, _SIM_DB_DIR):
        if directory and os.path.isdir(directory):
            candidates = sorted(
                f
                for f in os.listdir(directory)
                if f.startswith("fluxonium") and f.endswith((".h5", ".hdf5"))
            )
            if candidates:
                return os.path.join(directory, candidates[0])
    return ""
