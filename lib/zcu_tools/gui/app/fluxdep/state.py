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
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.gui.project import ProjectInfo
from zcu_tools.notebook.persistance import PointsData, SpectrumData, TransitionDict

logger = logging.getLogger(__name__)

# VersionTable is the shared optimistic-concurrency mechanism (app-agnostic);
# re-exported so ``state.VersionTable`` stays resolvable. fluxdep's key set
# (project / selection / spectrum:<name> / spectrums:__set__ / fit) is
# documented beside the *_VERSION_KEY constants below.
from zcu_tools.gui.version_table import (
    VersionTable as VersionTable,  # noqa: E402  (re-export)
)

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
    r_f: float | None,
    sample_f: float | None,
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

    selected: NDArray[np.bool_] | None = None
    min_distance: float = 0.0


@dataclass
class FitState:
    """Database-search fit inputs and result (the v2 pipeline tail).

    The inputs (``database_path`` / bounds / ``transitions`` / ``r_f`` /
    ``sample_f``) parameterise ``search_in_database``; the result
    (``params`` = (EJ, EC, EL)) is filled by a search. All of it is a
    process-lifetime singleton on State — one fit per session — so its version
    key (``fit``) is never dropped, only bumped.

    ``transitions`` is a ``TransitionDict`` (TypedDict, accessed with ``[...]``).
    """

    database_path: str = ""
    EJb: tuple[float, float] = (2.0, 15.0)
    ECb: tuple[float, float] = (0.2, 2.0)
    ELb: tuple[float, float] = (0.1, 2.0)
    transitions: TransitionDict = field(default_factory=default_transitions)
    # None means "not provided" (distinct from 0.0); a transition category that
    # needs one (blue/red side → r_f, mirror → sample_f) must have it set.
    r_f: float | None = None
    sample_f: float | None = None
    params: tuple[float, float, float] | None = None  # (EJ, EC, EL)

    @property
    def has_result(self) -> bool:
        return self.params is not None


class FluxDepState:
    """Passive GUI state container for the fluxdep analysis pipeline."""

    def __init__(self, project: ProjectInfo | None = None) -> None:
        self.project: ProjectInfo = project if project is not None else ProjectInfo()
        self.spectrums: dict[str, SpectrumEntry] = {}
        self.active_spectrum: str | None = None
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

    def set_active(self, name: str | None) -> None:
        if name is not None and name not in self.spectrums:
            raise KeyError(f"no spectrum named {name!r}")
        self.active_spectrum = name

    def set_alignment(
        self,
        name: str,
        flux_half: float,
        flux_int: float,
        flux_period: float,
        new_fluxs: NDArray[np.float64],
    ) -> None:
        """Record a spectrum's flux alignment and re-mapped flux axis.

        ``new_fluxs`` is the re-derived raw flux axis (computed by the caller,
        which owns the ``value2flux`` mapping). It is written in place on the
        entry's raw TypedDict here so every mutation of this spectrum — the
        alignment scalars and the flux axis — happens at this single State
        boundary under one version bump.
        """
        entry = self.spectrums[name]
        entry.raw["fluxs"] = np.asarray(new_fluxs, dtype=np.float64)
        self.spectrums[name] = replace(
            entry,
            flux_half=flux_half,
            flux_int=flux_int,
            flux_period=flux_period,
            aligned=True,
            alignment_seeded=True,
        )
        self.version.bump(spectrum_version_key(name))

    def set_points(self, name: str, points: PointsData) -> None:
        """Record a spectrum's selected points; mark selected iff non-empty.

        An empty point set is a legal outcome (the user deselected everything),
        but it must not be flagged ``points_selected`` — downstream readers
        (e.g. the cross-spectrum SelectorWidget) treat that flag as "has points
        to work with" and would crash on an empty cloud. Same ``freqs.size > 0``
        rule the load path uses (see ``LoadService``).
        """
        self.spectrums[name] = replace(
            self.spectrums[name],
            points=points,
            points_selected=points["freqs"].size > 0,
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
        r_f: float | None,
        sample_f: float | None,
    ) -> None:
        """Record the search inputs; clears any stale result.

        Changing the inputs invalidates a prior search result (it was for the old
        parameters), so ``params`` resets to None — a downstream reader never
        sees a result that disagrees with the inputs it reads.
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

    def set_fit_result(self, params: tuple[float, float, float]) -> None:
        """Record a search result onto the current fit inputs."""
        self.fit = replace(self.fit, params=params)
        self.version.bump(FIT_VERSION_KEY)
