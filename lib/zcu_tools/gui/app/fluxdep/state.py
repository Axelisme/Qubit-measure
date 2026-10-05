"""Passive state container for fluxdep-gui — shared by Controller and services.

Holds the analysis pipeline's state: the loaded-and-annotated spectrum
collection, the active spectrum, the cross-spectrum selection, and the
optimistic-concurrency ``VersionTable``. Like measure-gui, every State write
happens only on the Qt main thread; workers never mutate State directly (their
only side effect is emitting a Qt signal whose main-thread slot writes here).

``VersionTable`` is the shared GUI concurrency mechanism; the
domain shape (``FluxDepState`` / ``SpectrumEntry`` / ...) is fluxdep-specific and
replaces measure's tab/device/context model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.models import PointsData, TransitionDict
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.gui.owner import OwnerThreadGuard
from zcu_tools.gui.project import ProjectInfo

logger = logging.getLogger(__name__)

# VersionTable is the shared optimistic-concurrency mechanism (app-agnostic);
# re-exported so ``state.VersionTable`` stays resolvable. fluxdep's key set
# (project / selection / spectrum:<name> / spectrums:__set__ / fit) is
# documented beside the *_VERSION_KEY constants below.
from zcu_tools.gui.version_table import (
    VersionTable as VersionTable,  # noqa: E402  (re-export)
)

SpecType = Literal["OneTone", "TwoTone"]

# Spectrum leaf keys use retire on removal so reusable names do not repeat
# versions. Collection/global keys live for this State's lifetime.
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
    return any("mirror" in name and transitions.get(name) for name in transitions)


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
    out = transitions.copy()
    if r_f is not None:
        out["r_f"] = r_f
    if sample_f is not None:
        out["sample_f"] = sample_f
    return out


def spectrum_version_key(name: str) -> str:
    """Per-spectrum version key (``spectrum:<name>``)."""
    return f"spectrum:{name}"


@dataclass
class SpectrumEntry:
    """One loaded-and-annotated spectrum (≈ analysis.fluxdep.models.SpectrumResult + edit state).

    ``flux_half`` / ``flux_int`` / ``flux_period`` are per-spectrum: each spectrum
    is aligned on its own (the values may be *inherited* as an initial guess from
    an already-loaded spectrum, then fine-tuned). ``aligned`` / ``points_completed``
    gate the pipeline stage shown for this spectrum; completion includes zero
    points. name is the collection identifier; spec_type selects OneTone or
    TwoTone tooling. raw holds native device values, normalized flux, GHz
    frequency and complex signals; points holds annotated coordinates in the
    same units. flux_half/flux_int are native device positions of half/integer
    flux; flux_period is the device span of one flux quantum. aligned means
    alignment is committed; alignment_seeded means the scalars can seed a picker.
    """

    name: str
    spec_type: SpecType
    raw: SpectrumData  # dev_values / fluxs / freqs / signals(complex)
    points: PointsData  # selected points: dev_values / fluxs / freqs
    flux_half: float = 0.0
    flux_int: float = 0.0
    flux_period: float = 1.0
    aligned: bool = False
    points_completed: bool = False  # A result was committed, including zero points.
    # True when flux_half/int are a meaningful initial guess (inherited from
    # another spectrum or already aligned), so the line-picker should seed from
    # them rather than its centre/edge defaults.
    alignment_seeded: bool = False

    @property
    def point_count(self) -> int:
        """Number of available annotated points, independent of stage completion."""
        return int(self.points["freqs"].size)


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
    ``sample_f``) parameterise ``search_database``; the result
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
        self._owner_guard = OwnerThreadGuard()
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
        self._assert_owner()
        is_new = entry.name not in self.spectrums
        self.spectrums[entry.name] = entry
        self.version.bump(spectrum_version_key(entry.name))
        if is_new:
            self.version.bump(SPECTRUM_SET_VERSION_KEY)
        logger.debug("put_spectrum: name=%r new=%s", entry.name, is_new)

    def remove_spectrum(self, name: str) -> None:
        """Remove ``name`` and retire only its exact spectrum version key.

        Unknown names raise KeyError without mutation. The removed key reads
        as 0; reloading the same name continues above its previous live version.
        The collection version advances, and an active removed name is cleared.
        """
        self._assert_owner()
        del self.spectrums[name]
        self.version.retire(spectrum_version_key(name))
        self.version.bump(SPECTRUM_SET_VERSION_KEY)
        if self.active_spectrum == name:
            self.active_spectrum = None
        logger.debug("remove_spectrum: name=%r", name)

    def set_active(self, name: str | None) -> None:
        self._assert_owner()
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
        new_point_fluxs: NDArray[np.float64],
    ) -> None:
        """Commit alignment scalars and both mapped axes under one version bump.

        On the owner thread, replace raw fluxs with the finite 1-D
        ``new_fluxs`` matching raw dev_values, and point fluxs with the finite
        1-D ``new_point_fluxs`` matching native point dev_values. Preserve native
        points and completion. Unknown name raises KeyError; invalid mapped
        arrays raise ValueError before any mutation.
        """
        self._assert_owner()
        entry = self.spectrums[name]
        raw_fluxs = np.asarray(new_fluxs, dtype=np.float64)
        point_fluxs = np.asarray(new_point_fluxs, dtype=np.float64)
        if raw_fluxs.ndim != 1 or raw_fluxs.shape != entry.raw["dev_values"].shape:
            raise ValueError("new_fluxs must match raw dev_values shape")
        if (
            point_fluxs.ndim != 1
            or point_fluxs.shape != entry.points["dev_values"].shape
        ):
            raise ValueError("new_point_fluxs must match point dev_values shape")
        if not np.all(np.isfinite(raw_fluxs)) or not np.all(np.isfinite(point_fluxs)):
            raise ValueError("mapped fluxs must be finite")
        entry.raw["fluxs"] = raw_fluxs
        self.spectrums[name] = replace(
            entry,
            flux_half=flux_half,
            flux_int=flux_int,
            flux_period=flux_period,
            aligned=True,
            alignment_seeded=True,
            points=PointsData(
                dev_values=entry.points["dev_values"],
                fluxs=point_fluxs,
                freqs=entry.points["freqs"],
            ),
        )
        self.version.bump(spectrum_version_key(name))

    def set_points(self, name: str, points: PointsData) -> None:
        """Commit native/mapped points and complete picking, including zero points.

        Owner-thread command. Bump the named spectrum once; unknown name raises
        KeyError. Availability is point_count, not the completion flag.
        """
        self._assert_owner()
        self.spectrums[name] = replace(
            self.spectrums[name],
            points=points,
            points_completed=True,
        )
        self.version.bump(spectrum_version_key(name))

    def reset_points(self, name: str) -> None:
        """Clear points and completion on an aligned spectrum; bump once.

        Keep alignment and its seed. Unknown name raises KeyError; an unaligned
        spectrum raises ValueError. Both failures leave data/version unchanged.
        Must run on the owner thread.
        """
        self._assert_owner()
        entry = self.spectrums[name]
        if not entry.aligned:
            raise ValueError("spectrum must be aligned before resetting points")
        self.spectrums[name] = replace(
            entry,
            points=PointsData(
                dev_values=np.empty(0, dtype=np.float64),
                fluxs=np.empty(0, dtype=np.float64),
                freqs=np.empty(0, dtype=np.float64),
            ),
            points_completed=False,
        )
        self.version.bump(spectrum_version_key(name))

    def reset_alignment(self, name: str) -> None:
        """Reopen alignment, preserving native points, completion and last mapping.

        Set aligned=False and bump once. Unknown name raises KeyError without
        mutation. Must run on the owner thread.
        """
        self._assert_owner()
        self.spectrums[name] = replace(self.spectrums[name], aligned=False)
        self.version.bump(spectrum_version_key(name))

    def set_selection(
        self, selected: NDArray[np.bool_], min_distance: float = 0.0
    ) -> None:
        self._assert_owner()
        self.selection = SelectionState(selected=selected, min_distance=min_distance)
        self.version.bump(SELECTION_VERSION_KEY)

    def set_project(self, project: ProjectInfo) -> None:
        self._assert_owner()
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
        self._assert_owner()
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
        self._assert_owner()
        self.fit = replace(self.fit, params=params)
        self.version.bump(FIT_VERSION_KEY)

    def assert_owner_thread(self) -> None:
        """Reject foreign-thread access to owner-only snapshots and commands.

        The owner is the thread constructing this State. Raises RuntimeError
        without reading or changing any pipeline data.
        """
        self._owner_guard.assert_owner()

    def _assert_owner(self) -> None:
        self.assert_owner_thread()
