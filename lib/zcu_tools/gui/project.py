"""Shared project identity + default output-path derivation for the GUI apps.

Qt-free pure data: a ``ProjectInfo`` dataclass (where to read raw measurement
data from and where to write processed results) plus the ``default_*`` helpers
that derive the notebook-layout paths from a chip/qubit pair. Both fluxdep-gui
and dispersive-gui share this exact shape — they only ever differ in a UI label
string, which lives in the dialog, not here. App-agnostic and import-clean
(stdlib only): importing this module pulls in neither Qt nor matplotlib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Placeholder chip / qubit names used until the user sets a real project. They
# live here (the ProjectInfo's home) so both the export path and ProjectInfo's
# defaults share one source.
DEFAULT_CHIP = "unknown_chip"
DEFAULT_QUBIT = "unknown_qubit"


def nearest_existing(path: str) -> str:
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


def default_result_dir(chip_name: str, qub_name: str, root: str = "") -> str:
    """The notebook-layout result dir for a chip/qubit (``result/<chip>/<qubit>``).

    Empty names fall back to ``unknown_chip`` / ``unknown_qubit`` so the path is
    always well-formed. ``root`` (the repo root, injected by the entry script)
    anchors the result tree there instead of leaving it relative to cwd — a .bat
    launcher does ``cd /d "%~dp0"`` into script/, which would otherwise scope the
    default under script/. Empty ``root`` keeps the legacy cwd-relative form.
    """
    chip = chip_name or DEFAULT_CHIP
    qub = qub_name or DEFAULT_QUBIT
    return (
        os.path.join(root, "result", chip, qub)
        if root
        else os.path.join("result", chip, qub)
    )


def default_database_root(chip_name: str, qub_name: str, root: str = "") -> str:
    """The default *raw measurement* root for a chip/qubit (``Database/<chip>/<qubit>``).

    Raw measurement hdf5 files live under the repo's ``Database/`` tree per
    chip/qubit (measure-gui saves to ``Database/<chip>/<qubit>/<date>/...``, e.g.
    ``Database/Q5_2D/Q1/2026/05/Data_0504/R1_flux_2.hdf5``), NOT under ``result/``,
    which holds *processed* outputs (the exported ``spectrums.hdf5`` / params.json).
    Pointing a load dialog at ``result/`` made it fall back to the bare ``result``
    folder when the chip/qubit subdir had no raw data.

    Empty names fall back to ``unknown_chip`` / ``unknown_qubit`` so the path is
    always well-formed. ``root`` anchors it at the repo root (see
    ``default_result_dir``).
    """
    chip = chip_name or DEFAULT_CHIP
    qub = qub_name or DEFAULT_QUBIT
    return (
        os.path.join(root, "Database", chip, qub)
        if root
        else os.path.join("Database", chip, qub)
    )


@dataclass
class ProjectInfo:
    """Where to read raw measurement data from and where to write processed results.

    Locates files only — there is no chip/qub connection concept (these analysis
    GUIs never touch hardware). The chip / qubit names default to the
    ``unknown_*`` placeholders.

    ``result_dir`` and ``database_path`` are always concrete paths, never an empty
    sentinel: leave a field unset (empty) at construction and ``__post_init__``
    derives it from the chip/qubit names — so however a ``ProjectInfo`` is built,
    both paths come out well-formed and every save site can
    ``makedirs(exist_ok=True)`` a real directory without per-call-site fallback.
    Pass a non-empty value to *override* the derivation (the GUI does this when the
    user edits or browses the field). The empty placeholder never leaks out: it is
    resolved in ``__post_init__`` before the instance is observable.
    """

    chip_name: str = DEFAULT_CHIP
    qub_name: str = DEFAULT_QUBIT
    # Empty = "derive from chip/qubit in __post_init__"; a value overrides it.
    result_dir: str = ""  # → result/<chip>/<qubit>
    database_path: str = ""  # raw measurement root → Database/<chip>/<qubit>
    # Base dir the derived defaults are anchored under (the repo root, injected by
    # the entry script). Empty keeps the legacy cwd-relative default. Not a path
    # itself — only seeds the derivation below; the GUI re-derives via the dialog.
    root_dir: str = ""

    def __post_init__(self) -> None:
        # Single derivation point: an unset (empty) path becomes the chip/qubit
        # default (anchored at root_dir); a provided path is kept as the override.
        if not self.result_dir:
            self.result_dir = default_result_dir(
                self.chip_name, self.qub_name, self.root_dir
            )
        if not self.database_path:
            self.database_path = default_database_root(
                self.chip_name, self.qub_name, self.root_dir
            )


# ---------------------------------------------------------------------------
# Shared wire helpers used by read-only remote bridges (fluxdep, dispersive).
# ---------------------------------------------------------------------------


def project_info_payload(project: ProjectInfo) -> dict:
    """Build the 4-field wire payload for ``project.info`` handlers.

    Both fluxdep-gui and dispersive-gui expose an identical ``project.info``
    RPC that returns these four fields.  Centralising the mapping here ensures
    the two apps can never drift from each other.
    """
    return {
        "chip_name": project.chip_name,
        "qub_name": project.qub_name,
        "result_dir": project.result_dir,
        "database_path": project.database_path,
    }


def is_real_project(project: ProjectInfo) -> bool:
    """Return ``True`` when the project has been set to real chip/qubit names.

    The ``unknown_chip`` / ``unknown_qubit`` placeholders are the defaults a
    fresh ``ProjectInfo`` starts with; the user has not yet picked a project
    until both names are non-empty *and* neither matches the placeholder pair.
    This check is shared by both fluxdep-gui and dispersive-gui ``state.check``
    handlers so the definition of "has a real project" stays in one place.
    """
    return bool(
        project.chip_name
        and project.qub_name
        and (project.chip_name, project.qub_name) != (DEFAULT_CHIP, DEFAULT_QUBIT)
    )
