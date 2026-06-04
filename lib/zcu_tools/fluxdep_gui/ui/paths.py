"""Sensible default starting directories for the file dialogs.

Derived from the project (chip / qubit / result_dir / database_path) so the user
doesn't have to navigate from scratch each time. The *intended* directory is
computed from the project layout (e.g. ``result/<chip>/<qubit>/data/fluxdep`` for
processed spectra) even when it doesn't exist yet (before the first export); the
helper then returns the nearest existing ancestor, so the dialog opens close to
the right place instead of at the root.
"""

from __future__ import annotations

import os

from zcu_tools.fluxdep_gui.services.export import default_result_dir
from zcu_tools.fluxdep_gui.state import ProjectInfo

# The repo's bundled simulation databases (relative to the cwd a launch uses).
_SIM_DB_DIR = os.path.join("Database", "simulation")


def _nearest_existing(path: str) -> str:
    """The deepest existing ancestor of ``path`` (``path`` itself if it exists).

    Lets a dialog open near a not-yet-created target instead of at the cwd/root.
    Returns "" if nothing in the chain exists (then Qt opens the cwd).
    """
    path = os.path.abspath(path) if path else ""
    while path and not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:  # reached the filesystem root
            return ""
        path = parent
    return path


def _project_result_dir(project: ProjectInfo) -> str:
    """The project's result dir, or the chip/qubit-derived default if unset."""
    return project.result_dir or default_result_dir(project.chip_name, project.qub_name)


def raw_spectrum_dir(project: ProjectInfo) -> str:
    """Where raw spectrum hdf5 files live — the project's database_path root."""
    return _nearest_existing(project.database_path)


def processed_spectrum_dir(project: ProjectInfo) -> str:
    """Where exported spectrums.hdf5 lives — ``<result_dir>/data/fluxdep``."""
    target = os.path.join(_project_result_dir(project), "data", "fluxdep")
    return _nearest_existing(target)


def database_dir(project: ProjectInfo) -> str:
    """Where the precomputed search database lives.

    The project's database_path if set, else the repo's bundled
    ``Database/simulation`` directory.
    """
    if project.database_path:
        return _nearest_existing(project.database_path)
    return _nearest_existing(_SIM_DB_DIR)


def params_dir(project: ProjectInfo) -> str:
    """Where params.json is written — the project's result_dir."""
    return _nearest_existing(_project_result_dir(project))


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
