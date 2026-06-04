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
import os
from dataclasses import dataclass, field, replace
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray

from zcu_tools.notebook.persistance import PointsData, SpectrumData, TransitionDict

logger = logging.getLogger(__name__)

SpecType = Literal["OneTone", "TwoTone"]

# Version-table resource keys (see VersionTable docstring for the bump↔drop map).
SPECTRUM_SET_VERSION_KEY = "spectrums:__set__"
SELECTION_VERSION_KEY = "selection"
PROJECT_VERSION_KEY = "project"
FIT_VERSION_KEY = "fit"


def default_transitions() -> TransitionDict:
    """The default transition set (the notebook's common 'basic' choice).

    A fresh ``FitState`` starts here; the fit panel's preset dropdown swaps the
    whole dict and the form lets the user fine-tune each category. Frequencies
    (r_f / sample_f) are NOT stored here — they live on ``FitState`` directly.
    """
    return TransitionDict(
        {
            "transitions": [(0, 1), (0, 2), (1, 2), (1, 3)],
            "mirror": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)],
        }
    )


# Transition categories that require r_f / sample_f to be present (the model
# raises if a category here is used without the corresponding frequency key).
_R_F_CATEGORIES = ("blue side", "red side", "mirror blue", "mirror red")


def transitions_need_r_f(transitions: TransitionDict) -> bool:
    """Whether any present category needs ``r_f`` (blue/red side, mirror blue/red)."""
    return any(transitions.get(name) for name in _R_F_CATEGORIES)


def transitions_need_sample_f(transitions: TransitionDict) -> bool:
    """Whether any present category needs ``sample_f`` (anything with 'mirror')."""
    return any(
        "mirror" in name and transitions.get(name)
        for name in transitions
        if isinstance(name, str)
    )


def transitions_with_freqs(
    transitions: TransitionDict,
    r_f: Optional[float],
    sample_f: Optional[float],
) -> TransitionDict:
    """A copy of ``transitions`` with r_f / sample_f keys added when provided.

    The transition model keys on KEY PRESENCE (not value), and None means
    "unset", so only a provided frequency is injected. Callers should validate
    (via ``transitions_need_*``) that a needed frequency is present before search.
    """
    out: dict = dict(transitions)
    if r_f is not None:
        out["r_f"] = r_f
    if sample_f is not None:
        out["sample_f"] = sample_f
    return TransitionDict(out)  # type: ignore[arg-type]


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


# Placeholder chip / qubit names used until the user sets a real project. They
# live here (the ProjectInfo's home) so both the export path and ProjectInfo's
# defaults share one source; export.py re-exports them for back-compat.
DEFAULT_CHIP = "unknown_chip"
DEFAULT_QUBIT = "unknown_qubit"


def default_result_dir(chip_name: str, qub_name: str) -> str:
    """The notebook-layout result dir for a chip/qubit (``result/<chip>/<qubit>``).

    Empty names fall back to ``unknown_chip`` / ``unknown_qubit`` so the path is
    always well-formed.
    """
    chip = chip_name or DEFAULT_CHIP
    qub = qub_name or DEFAULT_QUBIT
    return os.path.join("result", chip, qub)


def default_database_root(chip_name: str, qub_name: str) -> str:
    """The default *raw spectrum* root for a chip/qubit.

    Raw measurement hdf5 files share the chip/qubit result tree, so this is the
    same ``result/<chip>/<qubit>`` directory. (This is the project's
    ``database_path``, distinct from the precomputed *search* database, whose
    default is the bundled ``Database/simulation`` — see ``ui/paths.database_dir``.)
    """
    return default_result_dir(chip_name, qub_name)


@dataclass
class ProjectInfo:
    """Where to read raw spectra from and where to write processed results.

    Locates files only — there is no chip/qub connection concept (fluxdep never
    touches hardware). A value-only block, so not versioned via a guarded op
    unless ``project.setup`` bumps ``project`` on replacement. The chip / qubit
    names default to the ``unknown_*`` placeholders.

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
    database_path: str = ""  # raw spectrum root → result/<chip>/<qubit>

    def __post_init__(self) -> None:
        # Single derivation point: an unset (empty) path becomes the chip/qubit
        # default; a provided path is kept as the user's override.
        if not self.result_dir:
            self.result_dir = default_result_dir(self.chip_name, self.qub_name)
        if not self.database_path:
            self.database_path = default_database_root(self.chip_name, self.qub_name)


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
    # True when flux_half/int are a meaningful initial guess (inherited from
    # another spectrum or already aligned), so the line-picker should seed from
    # them rather than its centre/edge defaults.
    alignment_seeded: bool = False


@dataclass
class SelectionState:
    """Cross-spectrum joint-point-cloud filtering (InteractiveSelector result).

    ``selected`` is a boolean mask over the joint point cloud assembled from all
    spectra's ``points``. The joint cloud itself (s_fluxs / s_freqs) is a derived
    value computed on query, not stored. ``min_distance`` is the downsample
    threshold (a stable filter parameter remembered across selector sessions —
    the brush selection itself is NOT remembered, so removed points are easy to
    bring back by re-opening with everything selected).
    """

    selected: Optional[NDArray[np.bool_]] = None
    min_distance: float = 0.0


@dataclass
class FitState:
    """Database-search fit inputs and result (the v2 pipeline tail).

    The inputs (``database_path`` / bounds / ``transitions`` / ``r_f`` /
    ``sample_f``) parameterise ``search_in_database``; the result
    (``params`` = (EJ, EC, EL) + ``best_dist``) is filled by a search. All of it
    is a process-lifetime singleton on State — one fit per session — so its
    version key (``fit``) is never dropped, only bumped.

    ``transitions`` is a ``TransitionDict`` (TypedDict, accessed with ``[...]``).
    """

    database_path: str = ""
    EJb: tuple[float, float] = (2.0, 15.0)
    ECb: tuple[float, float] = (0.2, 2.0)
    ELb: tuple[float, float] = (0.1, 2.0)
    transitions: TransitionDict = field(default_factory=default_transitions)
    # None means "not provided" (distinct from 0.0); a transition category that
    # needs one (blue/red side → r_f, mirror → sample_f) must have it set.
    r_f: Optional[float] = None
    sample_f: Optional[float] = None
    params: Optional[tuple[float, float, float]] = None  # (EJ, EC, EL)
    best_dist: Optional[float] = None

    @property
    def has_result(self) -> bool:
        return self.params is not None


class FluxDepState:
    """Passive GUI state container for the fluxdep analysis pipeline."""

    def __init__(self, project: Optional[ProjectInfo] = None) -> None:
        self.project: ProjectInfo = project if project is not None else ProjectInfo()
        self.spectrums: dict[str, SpectrumEntry] = {}
        self.active_spectrum: Optional[str] = None
        self.selection: SelectionState = SelectionState()
        self.fit: FitState = FitState()
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
            alignment_seeded=True,
        )
        self.version.bump(spectrum_version_key(name))

    def set_points(self, name: str, points: PointsData) -> None:
        """Record a spectrum's selected points and mark points selected."""
        self.spectrums[name] = replace(
            self.spectrums[name], points=points, points_selected=True
        )
        self.version.bump(spectrum_version_key(name))

    def set_selection(
        self, selected: NDArray[np.bool_], min_distance: float = 0.0
    ) -> None:
        self.selection = SelectionState(selected=selected, min_distance=min_distance)
        self.version.bump(SELECTION_VERSION_KEY)

    def set_project(self, project: ProjectInfo) -> None:
        self.project = project
        self.version.bump(PROJECT_VERSION_KEY)

    def set_fit_params(
        self,
        database_path: str,
        EJb: tuple[float, float],
        ECb: tuple[float, float],
        ELb: tuple[float, float],
        transitions: TransitionDict,
        r_f: Optional[float],
        sample_f: Optional[float],
    ) -> None:
        """Record the search inputs; clears any stale result.

        Changing the inputs invalidates a prior search result (it was for the old
        parameters), so ``params`` / ``best_dist`` reset to None — a downstream
        reader never sees a result that disagrees with the inputs it reads.
        """
        self.fit = FitState(
            database_path=database_path,
            EJb=EJb,
            ECb=ECb,
            ELb=ELb,
            transitions=transitions,
            r_f=r_f,
            sample_f=sample_f,
        )
        self.version.bump(FIT_VERSION_KEY)

    def set_fit_result(
        self, params: tuple[float, float, float], best_dist: float
    ) -> None:
        """Record a search result onto the current fit inputs."""
        self.fit = replace(self.fit, params=params, best_dist=best_dist)
        self.version.bump(FIT_VERSION_KEY)
