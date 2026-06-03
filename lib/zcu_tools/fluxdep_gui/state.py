"""Passive state container for fluxdep-gui — shared by Controller and services.

Holds the analysis pipeline's state: the loaded-and-annotated spectrum
collection, the active spectrum, the cross-spectrum selection, and the
optimistic-concurrency ``VersionTable``. Like measure-gui, every State write
happens only on the Qt main thread; workers never mutate State directly (their
only side effect is emitting a Qt signal whose main-thread slot writes here).

``VersionTable`` is copied verbatim from measure-gui (it is pure mechanism); the
domain shape (``FluxDepState`` / ``SpectrumEntry`` / ...) is fluxdep-specific and
replaces measure's tab/device/context model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from zcu_tools.notebook.persistance import PointsData, SpectrumData

logger = logging.getLogger(__name__)

SpecType = Literal["OneTone", "TwoTone"]

# Version-table resource keys (see VersionTable docstring for the bump↔drop map).
SPECTRUM_SET_VERSION_KEY = "spectrums:__set__"
SELECTION_VERSION_KEY = "selection"
PROJECT_VERSION_KEY = "project"


def spectrum_version_key(name: str) -> str:
    """Per-spectrum version key (``spectrum:<name>``)."""
    return f"spectrum:{name}"


class VersionTable:
    """Monotonic per-resource version counters (optimistic-concurrency guard).

    A passive container: each resource key maps to an integer that only ever
    increases by one per mutation. Callers (the resource-owning service, on the
    Qt main thread) ``bump`` a key when they actually write that resource's
    state. The guard compares an op's declared ``expected_versions`` against the
    current table atomically inside the main-thread dispatch sequence.

    Resource keys (fluxdep): ``project``, ``selection``, ``spectrum:<name>`` and
    ``spectrums:__set__`` (spectrum-set cardinality). A key absent from the table
    means version 0 (never bumped, or its resource was dropped — both read as
    "gone" by the guard).
    """

    def __init__(self) -> None:
        self._versions: dict[str, int] = {}

    def bump(self, key: str) -> int:
        """Advance a resource's version (a semantic write happened).

        PAIRING CONTRACT — every resource that gets bumped must be dropped by its
        owner's teardown, or a stale dependency would spuriously match a retained
        version. The bump↔drop map:
            spectrum:<name>    → SpectrumStore.remove_spectrum (drop_prefix spectrum:<name>)
            spectrums:__set__  → monotonic set-cardinality counter, never dropped
            selection          → process-lifetime singleton, never dropped
            project            → process-lifetime singleton, never dropped
        Adding a new bumped key means adding its drop to the owner's teardown.
        """
        new = self._versions.get(key, 0) + 1
        self._versions[key] = new
        logger.debug("version bump: %s -> %d", key, new)
        return new

    def get(self, key: str) -> int:
        """Current version of ``key`` (0 if never bumped / dropped)."""
        return self._versions.get(key, 0)

    def snapshot(self) -> dict[str, int]:
        """Full table copy (the ``resources.versions`` RPC payload)."""
        return dict(self._versions)

    def drop_prefix(self, prefix: str) -> None:
        """Forget every key starting with ``prefix`` (e.g. a removed spectrum).

        A dependency on a dropped key reads as version 0, which the guard treats
        as stale (the resource the caller depended on is gone).
        """
        doomed = [k for k in self._versions if k.startswith(prefix)]
        for k in doomed:
            del self._versions[k]
        if doomed:
            logger.debug("version drop_prefix: %s -> dropped %s", prefix, doomed)


@dataclass
class ProjectInfo:
    """Where to read raw spectra from and where to write processed results.

    Locates files only — there is no chip/qub connection concept (fluxdep never
    touches hardware). A value-only block, so not versioned via a guarded op
    unless ``project.setup`` bumps ``project`` on replacement.
    """

    chip_name: str = ""
    qub_name: str = ""
    result_dir: str = ""  # root for processed output (result_dir/data/fluxdep/...)
    database_path: str = ""  # root for raw spectrum hdf5 files


@dataclass
class SpectrumEntry:
    """One loaded-and-annotated spectrum (≈ persistance.SpectrumResult + edit state).

    ``flux_half`` / ``flux_int`` / ``flux_period`` are per-spectrum: each spectrum
    is aligned on its own (the values may be *inherited* as an initial guess from
    an already-loaded spectrum, then fine-tuned). ``aligned`` / ``points_selected``
    gate the pipeline stage shown for this spectrum.
    """

    name: str
    spec_type: SpecType
    raw: SpectrumData  # dev_values / fluxs / freqs / signals(complex)
    points: PointsData  # selected points: dev_values / fluxs / freqs
    flux_half: float = 0.0
    flux_int: float = 0.0
    flux_period: float = 1.0
    aligned: bool = False
    points_selected: bool = False


@dataclass
class SelectionState:
    """Cross-spectrum joint-point-cloud filtering (InteractiveSelector result).

    ``selected`` is a boolean mask over the joint point cloud assembled from all
    spectra's ``points``. The joint cloud itself (s_fluxs / s_freqs) is a derived
    value computed on query, not stored.
    """

    selected: Optional[NDArray[np.bool_]] = None


class FluxDepState:
    """Passive GUI state container for the fluxdep analysis pipeline."""

    def __init__(self, project: Optional[ProjectInfo] = None) -> None:
        self.project: ProjectInfo = project if project is not None else ProjectInfo()
        self.spectrums: dict[str, SpectrumEntry] = {}
        self.active_spectrum: Optional[str] = None
        self.selection: SelectionState = SelectionState()
        self.version = VersionTable()

    # ------------------------------------------------------------------
    # Spectrum collection (services write these on the Qt main thread).
    # ------------------------------------------------------------------

    def put_spectrum(self, entry: SpectrumEntry) -> None:
        """Insert or replace a spectrum entry.

        Bumps ``spectrum:<name>`` always; bumps ``spectrums:__set__`` only when
        the name is a *new* member (so a whole-set op such as ``export`` detects
        a concurrently-added spectrum; a re-load/replace of an existing name
        leaves the set cardinality unchanged).
        """
        is_new = entry.name not in self.spectrums
        self.spectrums[entry.name] = entry
        self.version.bump(spectrum_version_key(entry.name))
        if is_new:
            self.version.bump(SPECTRUM_SET_VERSION_KEY)
        logger.debug("put_spectrum: name=%r new=%s", entry.name, is_new)

    def remove_spectrum(self, name: str) -> None:
        """Remove a spectrum entry and drop its version keys."""
        del self.spectrums[name]
        self.version.drop_prefix(spectrum_version_key(name))
        self.version.bump(SPECTRUM_SET_VERSION_KEY)
        if self.active_spectrum == name:
            self.active_spectrum = None
        logger.debug("remove_spectrum: name=%r", name)

    def set_active(self, name: Optional[str]) -> None:
        if name is not None and name not in self.spectrums:
            raise KeyError(f"no spectrum named {name!r}")
        self.active_spectrum = name

    def set_alignment(
        self, name: str, flux_half: float, flux_int: float, flux_period: float
    ) -> None:
        """Record a spectrum's flux alignment and mark it aligned."""
        self.spectrums[name] = replace(
            self.spectrums[name],
            flux_half=flux_half,
            flux_int=flux_int,
            flux_period=flux_period,
            aligned=True,
        )
        self.version.bump(spectrum_version_key(name))

    def set_points(self, name: str, points: PointsData) -> None:
        """Record a spectrum's selected points and mark points selected."""
        self.spectrums[name] = replace(
            self.spectrums[name], points=points, points_selected=True
        )
        self.version.bump(spectrum_version_key(name))

    def set_selection(self, selected: NDArray[np.bool_]) -> None:
        self.selection = SelectionState(selected=selected)
        self.version.bump(SELECTION_VERSION_KEY)

    def set_project(self, project: ProjectInfo) -> None:
        self.project = project
        self.version.bump(PROJECT_VERSION_KEY)
