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

from zcu_tools.gui.app.fluxdep.state import ProjectInfo

# The repo's bundled simulation databases (relative to the cwd a launch uses).
# This is the home of the precomputed *search* database (fluxonium*.h5) — a
# shared resource, unrelated to a chip/qubit, so it is NOT derived from the
# project (cf. the project's database_path, which is the *raw spectrum* root).
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


def raw_spectrum_dir(project: ProjectInfo) -> str:
    """Where raw spectrum hdf5 files live — the project's database_path root."""
    return _nearest_existing(project.database_path)


def processed_spectrum_dir(project: ProjectInfo) -> str:
    """Where exported spectrums.hdf5 lives — ``<result_dir>/data/fluxdep``."""
    target = os.path.join(project.result_dir, "data", "fluxdep")
    return _nearest_existing(target)


def database_dir(project: ProjectInfo) -> str:
    """Where the precomputed *search* database lives — the bundled simulation dir.

    The search database (fluxonium*.h5) is a shared resource unrelated to the
    chip/qubit, so it is the repo's bundled ``Database/simulation`` — NOT derived
    from the project (whose ``database_path`` is the raw spectrum root).
    """
    del project  # search db location is project-independent
    return _nearest_existing(_SIM_DB_DIR)


def params_dir(project: ProjectInfo) -> str:
    """Where params.json is written — the project's result_dir."""
    return _nearest_existing(project.result_dir)


def default_database_file(project: ProjectInfo) -> str:
    """A pre-fillable default *search* database FILE, so the user often need not browse.

    Picks the first ``fluxonium*.h5`` in the bundled ``Database/simulation`` dir.
    Returns "" when none is found. (Project-independent — the search database is a
    shared resource, not a per-chip/qubit file; cf. ``database_dir``.)
    """
    del project  # search db location is project-independent
    if os.path.isdir(_SIM_DB_DIR):
        candidates = sorted(
            f
            for f in os.listdir(_SIM_DB_DIR)
            if f.startswith("fluxonium") and f.endswith((".h5", ".hdf5"))
        )
        if candidates:
            return os.path.join(_SIM_DB_DIR, candidates[0])
    return ""
