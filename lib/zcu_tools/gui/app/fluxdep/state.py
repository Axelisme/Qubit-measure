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
from zcu_tools.analysis.fluxdep.search import DatabaseSearchResult
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.owner import OwnerThreadGuard
from zcu_tools.gui.project import ProjectInfo

# VersionTable is the shared optimistic-concurrency mechanism (app-agnostic);
# re-exported so ``state.VersionTable`` stays resolvable. fluxdep's key set
# (project / selection / spectrum:<name> / spectrums:__set__ / fit) is
# documented beside the *_VERSION_KEY constants below.
from zcu_tools.gui.version_table import (
    VersionTable as VersionTable,  # (re-export)
)

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class AlignmentPublication:
    """Alignment values and provenance published with both mapped axes.

    flux_half and flux_int are half/integer positions in native device units.
    flux_period is the positive native-device span of one flux quantum.
    inherited_from is the spectrum name supplying untouched alignment, or None
    for directly/manual/auto-aligned values. State commits this carrier with its
    raw and point flux arrays under a single spectrum version bump.
    """

    flux_half: float
    flux_int: float
    flux_period: float
    inherited_from: str | None = None


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
    # Source of the published alignment, or None for direct/manual alignment.
    inherited_from: str | None = None
    # None means no Filter (all selected). Otherwise, a bool array of length
    # point_count, indexed by this spectrum's points, not joint-cloud positions.
    filter_mask: NDArray[np.bool_] | None = None

    @property
    def point_count(self) -> int:
        """Number of available annotated points, independent of stage completion."""
        return int(self.points["freqs"].size)


@dataclass
class SelectionState:
    """Stable downsample threshold remembered across selector sessions.

    Filter masks live on each SpectrumEntry. The brush selection in a newly
    opened selector is all-selected, independent of the published masks.
    """

    min_distance: float = 0.0


@dataclass(frozen=True)
class PipelineStage:
    """One captured node in the app's fit dependency graph.

    name is an internal unique node ID, not a spectrum identifier. version_key
    names a counter in State's VersionTable; version records its captured value.
    inputs names direct upstream nodes in the same snapshot. label is the
    human-readable downstream invalidation target; reason describes a changed
    input. Multiple per-spectrum mask nodes reuse the global selection counter.
    """

    name: str
    version_key: str
    version: int
    inputs: tuple[str, ...]
    label: str
    reason: str


@dataclass
class FitState:
    """Database-search fit inputs and result (the v2 pipeline tail).

    The inputs (``search_db`` / bounds / ``transitions`` / ``r_f`` /
    ``sample_f``) parameterise ``search_database``; the result
    (``params`` = (EJ, EC, EL)) is filled by a search. All of it is a
    process-lifetime singleton on State — one fit per session — so its version
    key (``fit``) is never dropped, only bumped.

    ``transitions`` is a ``TransitionDict`` (TypedDict, accessed with ``[...]``).
    """

    search_db: str = ""
    EJb: tuple[float, float] = (2.0, 15.0)
    ECb: tuple[float, float] = (0.2, 2.0)
    ELb: tuple[float, float] = (0.1, 2.0)
    transitions: TransitionDict = field(default_factory=default_transitions)
    # None means "not provided" (distinct from 0.0); a transition category that
    # needs one (blue/red side → r_f, mirror → sample_f) must have it set.
    r_f: float | None = None
    sample_f: float | None = None
    params: tuple[float, float, float] | None = None  # (EJ, EC, EL)

    # Recorded kernel diagnostics/attribution; None for params-only publication.
    result: DatabaseSearchResult | None = None
    # Captured graph/counters after result publication; empty without a result.
    input_stages: tuple[PipelineStage, ...] = ()

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
        leaves the set cardinality unchanged). Discards this spectrum's Filter
        mask; bumps ``SELECTION_VERSION_KEY`` if the stored entry had a mask.
        The incoming entry must have no Filter mask. Otherwise raises ValueError
        before changing state or versions. Stores the caller's entry directly,
        without changing it.
        """
        self._assert_owner()
        if entry.filter_mask is not None:
            raise ValueError("a new spectrum entry cannot carry a Filter mask")
        is_new = entry.name not in self.spectrums
        if not is_new:
            self._discard_filter_mask(entry.name)
        self.spectrums[entry.name] = entry
        self.version.bump(spectrum_version_key(entry.name))
        self.version.bump(f"__pipeline:alignment:{entry.name}")
        self.version.bump(f"__pipeline:points:{entry.name}")
        if is_new:
            self.version.bump(SPECTRUM_SET_VERSION_KEY)
        logger.debug("put_spectrum: name=%r new=%s", entry.name, is_new)

    def get_spectrum(self, name: str) -> SpectrumEntry:
        """Return the live entry for an opaque literal name, on the owner thread.

        The returned entry is not copied. Unknown names raise InvalidInputError
        with reason unknown_spectrum; foreign threads raise RuntimeError.
        Neither failure changes State or versions.
        """
        self._assert_owner()
        try:
            return self.spectrums[name]
        except KeyError as exc:
            raise InvalidInputError(
                f"no spectrum named {name!r}", reason_code="unknown_spectrum"
            ) from exc

    def remove_spectrum(self, name: str) -> None:
        """Remove ``name`` and retire only its exact spectrum version key.

        Unknown names raise InvalidInputError (unknown_spectrum) without mutation.
        The removed key reads as 0; reloading the same name continues above its
        previous live version.
        The collection version advances, and an active removed name is cleared.
        Discards this spectrum's Filter mask; bumps ``SELECTION_VERSION_KEY``
        if the entry had a mask.
        """
        self._assert_owner()
        self.get_spectrum(name)
        self._discard_filter_mask(name)
        del self.spectrums[name]
        self.version.retire(spectrum_version_key(name))
        self.version.bump(SPECTRUM_SET_VERSION_KEY)
        if self.active_spectrum == name:
            self.active_spectrum = None
        logger.debug("remove_spectrum: name=%r", name)

    def set_active(self, name: str | None) -> None:
        """Select an existing literal name, or clear selection with None.

        Owner-thread command. Unknown names raise InvalidInputError
        (unknown_spectrum) without changing active selection or versions.
        """
        self._assert_owner()
        if name is not None:
            self.get_spectrum(name)
        self.active_spectrum = name

    def set_alignment(
        self,
        name: str,
        alignment: AlignmentPublication,
        new_fluxs: NDArray[np.float64],
        new_point_fluxs: NDArray[np.float64],
    ) -> None:
        """Commit alignment scalars and both mapped axes under one version bump.

        On the owner thread, replace raw fluxs with the finite 1-D
        ``new_fluxs`` matching raw dev_values, and point fluxs with the finite
        1-D ``new_point_fluxs`` matching native point dev_values. Preserve native
        points and completion. alignment carries scalars and current source.
        Unknown name raises KeyError; invalid mapped
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
            flux_half=alignment.flux_half,
            flux_int=alignment.flux_int,
            flux_period=alignment.flux_period,
            inherited_from=alignment.inherited_from,
            aligned=True,
            alignment_seeded=True,
            points=PointsData(
                dev_values=entry.points["dev_values"],
                fluxs=point_fluxs,
                freqs=entry.points["freqs"],
            ),
        )
        self.version.bump(spectrum_version_key(name))
        self.version.bump(f"__pipeline:alignment:{name}")

    def set_points(self, name: str, points: PointsData) -> None:
        """Commit native/mapped points and complete picking, including zero points.

        Owner-thread command. Bumps ``spectrum:<name>`` and discards this
        spectrum's Filter mask. Bumps ``SELECTION_VERSION_KEY`` if it had a mask.
        Unknown name raises KeyError. Availability is point_count, not the
        completion flag.
        """
        self._assert_owner()
        self._discard_filter_mask(name)
        self.spectrums[name] = replace(
            self.spectrums[name],
            points=points,
            points_completed=True,
        )
        self.version.bump(spectrum_version_key(name))
        self.version.bump(f"__pipeline:points:{name}")

    def reset_points(self, name: str) -> None:
        """Clear points and completion on an aligned spectrum.

        Keep alignment and its seed. Bumps ``spectrum:<name>`` and discards this
        spectrum's Filter mask. Bumps ``SELECTION_VERSION_KEY`` if it had a mask.
        Unknown name raises InvalidInputError
        (unknown_spectrum); unaligned raises FailedPreconditionError
        (spectrum_not_aligned). Both leave data/version unchanged.
        Must run on the owner thread.
        """
        self._assert_owner()
        entry = self.get_spectrum(name)
        if not entry.aligned:
            raise FailedPreconditionError(
                "spectrum must be aligned before resetting points",
                reason_code="spectrum_not_aligned",
            )
        self._discard_filter_mask(name)
        self.spectrums[name] = replace(
            self.spectrums[name],
            points=PointsData(
                dev_values=np.empty(0, dtype=np.float64),
                fluxs=np.empty(0, dtype=np.float64),
                freqs=np.empty(0, dtype=np.float64),
            ),
            points_completed=False,
        )
        self.version.bump(spectrum_version_key(name))
        self.version.bump(f"__pipeline:points:{name}")

    def reset_alignment(self, name: str) -> None:
        """Reopen alignment, preserving native points, completion and last mapping.

        Set aligned=False and bump once. Unknown name raises InvalidInputError
        (unknown_spectrum) without mutation. Must run on the owner thread.
        """
        self._assert_owner()
        self.spectrums[name] = replace(self.get_spectrum(name), aligned=False)
        self.version.bump(spectrum_version_key(name))
        self.version.bump(f"__pipeline:alignment:{name}")

    def _discard_filter_mask(self, name: str) -> None:
        """Drop a stored spectrum's mask and advance selection only if present."""
        entry = self.spectrums[name]
        if entry.filter_mask is not None:
            self.version.bump(SELECTION_VERSION_KEY)
            self.spectrums[name] = replace(entry, filter_mask=None)

    def set_selection(
        self, selected: NDArray[np.bool_], min_distance: float = 0.0
    ) -> None:
        """Publish a joint-cloud Filter mask and remember its downsample threshold.

        ``selected`` must be a one-dimensional bool mask with one value per
        annotated point across all spectra, in insertion order. Split it into
        detached per-spectrum masks indexed by each spectrum's points.
        ``min_distance`` is the normalized downsample threshold for the next
        selector session. Bumps ``SELECTION_VERSION_KEY`` on publication.
        Raises ValueError for a shape mismatch before changing data or versions.
        Must run on the owner thread.
        """
        self._assert_owner()
        mask = np.asarray(selected, dtype=np.bool_)
        point_count = sum(entry.point_count for entry in self.spectrums.values())
        if mask.ndim == 1 and mask.size != point_count:
            raise ValueError(
                f"selection mask length {mask.size} != joint point cloud "
                f"size {point_count}"
            )
        if mask.shape != (point_count,):
            raise ValueError("selection mask must match the joint point cloud shape")
        offset = 0
        for name, entry in self.spectrums.items():
            end = offset + entry.point_count
            self.spectrums[name] = replace(entry, filter_mask=mask[offset:end].copy())
            offset = end
        self.selection = SelectionState(min_distance=min_distance)
        self.version.bump(SELECTION_VERSION_KEY)

    def selection_mask(self) -> NDArray[np.bool_] | None:
        """Project per-spectrum Filter masks in joint-cloud insertion order.

        Return a detached mask, with unfiltered spectra all-selected. None
        means no spectrum has a published mask. Alignment preserves indices;
        publishing points, replacing or removing a spectrum drops its mask.
        """
        self._assert_owner()
        entries = self.spectrums.values()
        if not any(entry.filter_mask is not None for entry in entries):
            return None
        return np.concatenate(
            [
                entry.filter_mask
                if entry.filter_mask is not None
                else np.ones(entry.point_count, dtype=np.bool_)
                for entry in entries
            ]
        )

    def set_project(self, project: ProjectInfo) -> None:
        self._assert_owner()
        self.project = project
        self.version.bump(PROJECT_VERSION_KEY)

    def set_fit_params(
        self,
        search_db: str,
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
            search_db=search_db,
            EJb=EJb,
            ECb=ECb,
            ELb=ELb,
            transitions=transitions,
            r_f=r_f,
            sample_f=sample_f,
        )
        self.version.bump(FIT_VERSION_KEY)

    def set_fit_result(
        self,
        params: tuple[float, float, float],
        *,
        result: DatabaseSearchResult | None = None,
    ) -> None:
        """Record EJ/EC/EL GHz and optional kernel diagnostics on current inputs.

        result carries recorded attribution and quality, never synthesized values.
        None publishes params only, so consumers must report absent quality.
        Owner-thread only; advances the existing fit version once.
        """
        self._assert_owner()
        self.fit = replace(self.fit, params=params, result=result)
        self.version.bump(FIT_VERSION_KEY)
        self.fit = replace(self.fit, input_stages=self.pipeline_snapshot())

    def pipeline_snapshot(self) -> tuple[PipelineStage, ...]:
        """Capture the single alignment → points → Filter → fit dependency graph.

        Reuses fit, collection and selection counters. Alignment and points have
        private stage counters because spectrum counters also guard contexts.
        Includes zero-point sources. Owner-thread only; does not mutate State.
        """
        self._assert_owner()
        stages: list[PipelineStage] = []
        masks: list[str] = []
        for name in self.spectrums:
            alignment = f"__pipeline:alignment:{name}"
            points = f"__pipeline:points:{name}"
            mask = f"__pipeline:mask:{name}"
            stages.extend(
                (
                    PipelineStage(
                        alignment,
                        alignment,
                        self.version.get(alignment),
                        (),
                        f"alignment of {name}",
                        f"{name} re-aligned",
                    ),
                    PipelineStage(
                        points,
                        points,
                        self.version.get(points),
                        (alignment,),
                        f"points of {name}",
                        f"points of {name} changed",
                    ),
                    PipelineStage(
                        mask,
                        SELECTION_VERSION_KEY,
                        self.version.get(SELECTION_VERSION_KEY),
                        (points,),
                        f"filter mask of {name}",
                        "filter changed",
                    ),
                )
            )
            masks.append(mask)
        if not masks:
            masks.append("__pipeline:mask")
            stages.append(
                PipelineStage(
                    "__pipeline:mask",
                    SELECTION_VERSION_KEY,
                    self.version.get(SELECTION_VERSION_KEY),
                    (),
                    "filter",
                    "filter changed",
                )
            )
        stages.append(
            PipelineStage(
                SPECTRUM_SET_VERSION_KEY,
                SPECTRUM_SET_VERSION_KEY,
                self.version.get(SPECTRUM_SET_VERSION_KEY),
                (),
                "spectrum collection",
                "spectrum collection changed",
            )
        )
        stages.append(
            PipelineStage(
                FIT_VERSION_KEY,
                FIT_VERSION_KEY,
                self.version.get(FIT_VERSION_KEY),
                (SPECTRUM_SET_VERSION_KEY, *masks),
                "fit",
                "fit parameters changed",
            )
        )
        return tuple(stages)

    def invalidated(self, before: tuple[PipelineStage, ...]) -> list[str]:
        """Return downstream labels made invalid since an owner snapshot.

        before comes from pipeline_snapshot immediately before a publication.
        The publishing node is not itself downstream. Returns graph order,
        deduplicated, including transitive points/mask/fit dependencies.
        Does not clear results or mutate State.
        """
        self._assert_owner()
        previous = {stage.name: stage.version for stage in before}
        affected: set[str] = set()
        labels: list[str] = []
        for stage in self.pipeline_snapshot():
            upstream_changed = any(name in affected for name in stage.inputs)
            changed = previous.get(stage.name, stage.version) != stage.version
            if upstream_changed:
                labels.append(stage.label)
            if changed or upstream_changed:
                affected.add(stage.name)
        return list(dict.fromkeys(labels))

    @property
    def fit_stale_reasons(self) -> list[str]:
        """Changed search inputs from the recorded graph, or [] without a result.

        Reasons describe root changes, not every transitively invalid node.
        Reads current counters against the recorded graph, so removals and
        same-name reloads cannot make an old result fresh. Active is not an input.
        """
        self._assert_owner()
        affected: set[str] = set()
        reasons: list[str] = []
        for stage in self.fit.input_stages:
            upstream_changed = any(name in affected for name in stage.inputs)
            changed = self.version.get(stage.version_key) != stage.version
            if changed and not upstream_changed:
                reasons.append(stage.reason)
            if changed or upstream_changed:
                affected.add(stage.name)
        return list(dict.fromkeys(reasons))

    def resource_versions(self) -> dict[str, int]:
        """Return existing public counters, excluding private pipeline stage keys."""
        self._assert_owner()
        return {
            key: value
            for key, value in self.version.snapshot().items()
            if not key.startswith("__pipeline:")
        }

    def assert_owner_thread(self) -> None:
        """Reject foreign-thread access to owner-only snapshots and commands.

        The owner is the thread constructing this State. Raises RuntimeError
        without reading or changing any pipeline data.
        """
        self._owner_guard.assert_owner()

    def _assert_owner(self) -> None:
        self.assert_owner_thread()
