"""Fluxdep-owned interactive contexts, independent of Qt presentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.cross_selection import (
    CrossSelectionBackground,
    CrossSelectionInputs,
    CrossSelectionResult,
    CrossSelectionState,
    analyze_cross_selection,
)
from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickAnalysis,
    FluxPickInputs,
    FluxPickState,
    analyze_flux_pick,
    fold_initial_lines,
)
from zcu_tools.analysis.fluxdep.onetone import (
    OneToneInputs,
    OneTonePickResult,
    OneTonePickState,
)
from zcu_tools.analysis.fluxdep.processing import cast2real_and_norm
from zcu_tools.analysis.fluxdep.twotone import (
    TwoToneInputs,
    TwoTonePickResult,
    TwoTonePickState,
)
from zcu_tools.gui.app.fluxdep.cross_selection import CrossSelectionPlugin
from zcu_tools.gui.app.fluxdep.event_bus import (
    ActiveSpectrumChangedPayload,
    EventBus,
    InteractiveChangedPayload,
    InteractiveKind,
    SelectionChangedPayload,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.gui.app.fluxdep.onetone import OneTonePickPlugin
from zcu_tools.gui.app.fluxdep.state import (
    SELECTION_VERSION_KEY,
    SPECTRUM_SET_VERSION_KEY,
    FluxDepState,
    SpectrumEntry,
    spectrum_version_key,
)
from zcu_tools.gui.app.fluxdep.twotone import TwoTonePickPlugin
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.interactive.flux_pick import SharedFluxPickPlugin
from zcu_tools.gui.interactive.plugin import BackgroundSubmitter
from zcu_tools.gui.interactive.session import Session
from zcu_tools.gui.session.ports import OwnerScheduler


@dataclass(frozen=True, slots=True)
class _NormalizedBackground:
    """Cached normalization tied to a spectrum version and raw signals object."""

    spectrum_version: int
    raw_signals: NDArray[np.complex128]
    normalized_signals: NDArray[np.float64]


@dataclass(slots=True)
class LinePickContext:
    """Live context owned by FluxDepInteractiveOwner, not by its widgets.

    spectrum_name identifies the literal spectrum whose captured arrays are used.
    plugin supplies shared device-axis commands and the accepted alignment result.
    session is the sole committed state; old contexts retain readable snapshots
    after cancellation but reject writes.
    """

    spectrum_name: str
    plugin: SharedFluxPickPlugin[FluxPickAnalysis]
    session: Session[FluxPickState]
    # Published source captured at begin; draft edits never mutate this source.
    inherited_from: str | None = None
    # Any accepted edit or auto-align start makes this draft independently aligned.
    alignment_changed: bool = False


@dataclass(frozen=True, slots=True)
class OneTonePickContext:
    """Live one-tone context with captured calibration references.

    spectrum_name identifies the aligned OneTone spectrum.
    plugin supplies threshold actions/commands and native point results.
    session owns complete committed threshold/indices and single-level undo.
    flux_half and flux_int are captured, finite native device reference lines,
    not editable calibration or a second analysis state.
    """

    spectrum_name: str
    plugin: OneTonePickPlugin
    session: Session[OneTonePickState]
    flux_half: float
    flux_int: float


@dataclass(frozen=True, slots=True)
class TwoTonePickContext:
    """Live TwoTone selection owned by the app rather than presentation.

    spectrum_name identifies the aligned TwoTone spectrum. plugin owns
    captured numerical inputs/actions; session holds committed mask/settings/
    tools and single-level undo. flux_half/flux_int are captured native device
    reference lines, not an editable or second calibration.
    """

    spectrum_name: str
    plugin: TwoTonePickPlugin
    session: Session[TwoTonePickState]
    flux_half: float
    flux_int: float


@dataclass(slots=True)
class CrossSelectionContext:
    """Live full-cloud selection owned by the app, never a fake spectrum.

    plugin owns captured numerical inputs and Actions; session is the sole
    committed mask/distance/tool with single Undo. source_versions contains all
    insertion-ordered (name, version) facts, including zero-point entries.
    spectrum_set_version captures the collection identity. selection_version is
    the captured publication version, updated only after this context's Apply.
    source_point_counts preserves each source's captured cloud size for retained
    result projection, including zero-point sources. Views may detach; closed
    old sessions reject writes but retain snapshots.
    """

    plugin: CrossSelectionPlugin
    session: Session[CrossSelectionState]
    source_versions: tuple[tuple[str, int], ...]
    # Captured insertion-order sizes map retained results to their source names.
    source_point_counts: tuple[tuple[str, int], ...]
    spectrum_set_version: int
    selection_version: int


InteractiveContext = (
    LinePickContext | OneTonePickContext | TwoTonePickContext | CrossSelectionContext
)


@dataclass(frozen=True, slots=True)
class _BeginCheckpoint:
    """Owner admission captured before the first invalidating context query.

    next_context_id is the existing identity allocation frontier, not a Session
    edit revision. source_versions records collection, selection and spectrum
    resource keys with their versions. Display selection is not an input guard.
    """

    next_context_id: int
    source_versions: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class ActiveInteractiveContext:
    """Live owner reference, not a detached numerical snapshot.

    context_id is a positive monotonic identity within one owner lifetime.
    Reused begin preserves it; retirement never permits its reuse.
    context contains the domain plugin and authoritative Session. Its input may
    later close while snapshots remain readable. Access requires the owner loop.
    """

    context_id: int
    context: InteractiveContext

    @property
    def kind(self) -> InteractiveKind:
        """Domain picker kind: line, onetone, twotone, or joint selection."""
        if isinstance(self.context, LinePickContext):
            return "line"
        if isinstance(self.context, OneTonePickContext):
            return "onetone"
        if isinstance(self.context, TwoTonePickContext):
            return "twotone"
        return "selection"

    @property
    def spectrum_name(self) -> str | None:
        """Picker source identity, or None for the joint-cloud selection."""
        return (
            None
            if isinstance(self.context, CrossSelectionContext)
            else self.context.spectrum_name
        )


@dataclass(frozen=True, slots=True)
class PublicationReceipt[T]:
    """One successful app publication, independent of subsequent owner changes.

    result is the accepted kernel numeric result for this command; its arrays
    remain detached from State. invalidated lists downstream labels captured by
    Controller before emitting the publication fact. Four terminal picker kinds
    share this atomic correspondence. Failures raise instead of returning a
    receipt; line/sample failures after retirement do not reopen input.
    """

    result: T
    invalidated: list[str]


@dataclass(frozen=True, slots=True)
class FluxDepInteractivePorts:
    """Runtime collaborators for app-owned picking and joint-cloud selection.

    background delivers computation on the owner loop; None rejects auto-align.
    publish_alignment receives spectrum name, native device half/integer lines,
    and the current inherited source (None for an edited or auto-aligned draft).
    publish_points receives name and native device/GHz arrays; PointsService owns
    sorting/calibration. derive_pointcloud queries the complete calibrated
    flux/GHz cloud in spectrum insertion order. publish_selection receives that
    cloud's full bool kept mask and normalized distance; Controller owns its
    single version bump/fact. Publication errors propagate without rollback.
    """

    background: BackgroundSubmitter | None
    publish_alignment: Callable[[str, float, float, str | None], list[str]]
    publish_points: Callable[[str, NDArray[np.float64], NDArray[np.float64]], list[str]]
    derive_pointcloud: Callable[[], tuple[NDArray[np.float64], NDArray[np.float64]]]
    publish_selection: Callable[[NDArray[np.bool_], float], list[str]]


class FluxDepInteractiveOwner:
    """Own one picker per spectrum and one global Filter on the owner loop.

    Views and active display may change without retiring drafts. Replacing a
    spectrum's picker kind closes only that spectrum's old input. Source
    publications retire affected pickers and the captured Filter cloud. Each
    line plugin has its own single-flight worker; late closed-input delivery
    cannot publish. dispose retires every draft and releases subscriptions.
    """

    def __init__(
        self,
        state: FluxDepState,
        bus: EventBus,
        owner: OwnerScheduler,
        *,
        ports: FluxDepInteractivePorts,
    ) -> None:
        """Capture State, bus and typed runtime ports on the supplied owner loop.

        ports defines background delivery, read ordering and Controller-owned
        publication contracts. See FluxDepInteractivePorts for units/array shapes.
        State is not copied; captured analysis inputs are copied when beginning.
        Off-owner construction/access raises RuntimeError. Views do not own input.
        """
        self._state = state
        self._bus = bus
        self._owner = owner
        self._require_owner()
        self._background = ports.background
        self._publish_alignment = ports.publish_alignment
        self._publish_points = ports.publish_points
        self._derive_pointcloud = ports.derive_pointcloud
        self._publish_selection = ports.publish_selection
        self._contexts: dict[str | None, ActiveInteractiveContext] = {}
        self._view_selection = False
        # Leaving Filter cancels its context, so each return would otherwise redo the
        # wavelet normalization of every unchanged spectrum.
        self._normalized_backgrounds: dict[str, _NormalizedBackground] = {}
        self._next_context_id = 1
        self._context_subscriptions: dict[int, tuple[Callable[[], None], ...]] = {}
        self._disposed = False
        self._publishing_selection = False
        self._subscriptions: tuple[Callable[[], None], ...] = (
            bus.subscribe(
                ActiveSpectrumChangedPayload, self._on_active_changed
            ).unsubscribe,
            bus.subscribe(SpectrumAddedPayload, self._on_spectrum_changed).unsubscribe,
            bus.subscribe(
                SelectionChangedPayload, self._on_selection_changed
            ).unsubscribe,
            bus.subscribe(
                SpectrumRemovedPayload, self._on_spectrum_changed
            ).unsubscribe,
            bus.subscribe(
                SpectrumChangedPayload, self._on_spectrum_changed
            ).unsubscribe,
        )

    def _require_owner(self) -> None:
        if not self._owner.is_owner_thread():
            raise RuntimeError("interactive owner access must run on the owner loop")

    def _target_key(self, name: str | None) -> str | None:
        if name == "selection":
            return None
        return (
            name
            if name is not None
            else (None if self._view_selection else self._state.active_spectrum)
        )

    def inspect(self, name: str | None = None) -> ActiveInteractiveContext | None:
        """Read a named live draft without changing display, input or Undo.

        name="selection" addresses the global Filter. Omitted name reads the
        displayed picker for GUI callers. IDs are owner-lifetime identities.
        Invalidated or closed input is retired; reads never establish a draft.
        """
        self._require_owner()
        key = self._target_key(name)
        active = self._contexts.get(key)
        if active is None:
            return None
        context = active.context
        if isinstance(context, CrossSelectionContext):
            if self.current_cross_selection() is None:
                return None
        elif context.spectrum_name not in self._state.spectrums:
            self.cancel(context.spectrum_name)
            return None
        elif isinstance(context, TwoTonePickContext):
            entry = self._state.spectrums[context.spectrum_name]
            if entry.spec_type != "TwoTone" or not entry.aligned:
                self.cancel(context.spectrum_name)
                return None
        try:
            context.session.ensure_input_open()
        except FailedPreconditionError:
            self.cancel("selection" if key is None else key)
            return None
        return active

    def _capture_begin_checkpoint(self) -> _BeginCheckpoint:
        """Capture admission before a query or retirement can notify callbacks."""
        source_keys = (
            SPECTRUM_SET_VERSION_KEY,
            SELECTION_VERSION_KEY,
            *(spectrum_version_key(name) for name in self._state.spectrums),
        )
        return _BeginCheckpoint(
            self._next_context_id,
            tuple((key, self._state.version.get(key)) for key in source_keys),
        )

    def _check_begin_checkpoint(self, checkpoint: _BeginCheckpoint) -> None:
        """Reject callback-created input, disposal or changed captured sources."""
        if (
            self._disposed
            or self._next_context_id != checkpoint.next_context_id
            or any(
                self._state.version.get(key) != version
                for key, version in checkpoint.source_versions
            )
        ):
            raise FailedPreconditionError("owner changed during context retirement")

    def _cancel_for_begin(self, checkpoint: _BeginCheckpoint, name: str) -> None:
        """Retire input, then check the same admission captured before queries."""
        self.cancel(name)
        self._check_begin_checkpoint(checkpoint)

    def _install_context(self, context: InteractiveContext) -> None:
        active = ActiveInteractiveContext(self._next_context_id, context)
        self._next_context_id += 1
        self._contexts[active.spectrum_name] = active
        self._view_selection = isinstance(context, CrossSelectionContext)
        subscriptions = [
            context.session.subscribe(lambda: self._session_updated(active))
        ]
        if isinstance(context, LinePickContext):
            subscriptions.append(
                context.plugin.subscribe_alignment(
                    lambda busy, _error: self._alignment_updated(active, busy=busy)
                )
            )
        self._context_subscriptions[active.context_id] = tuple(subscriptions)
        self._emit_interactive(active, "opened")

    def _session_updated(self, active: ActiveInteractiveContext) -> None:
        if isinstance(active.context, LinePickContext):
            active.context.alignment_changed = True
        self._updated(active)

    def _alignment_updated(
        self, active: ActiveInteractiveContext, *, busy: bool
    ) -> None:
        if isinstance(active.context, LinePickContext) and busy:
            active.context.alignment_changed = True
        self._updated(active)

    def _updated(self, active: ActiveInteractiveContext) -> None:
        if self._contexts.get(active.spectrum_name) is active:
            self._emit_interactive(active, "updated")

    def _emit_interactive(
        self,
        active: ActiveInteractiveContext,
        phase: Literal["opened", "updated", "closed"],
    ) -> None:
        self._bus.emit(
            InteractiveChangedPayload(
                context_id=active.context_id,
                kind=active.kind,
                spectrum_name=active.spectrum_name,
                phase=phase,
            )
        )

    def _on_active_changed(self, _event: ActiveSpectrumChangedPayload) -> None:
        # Only the displayed picker changes; all draft identities remain live.
        self._view_selection = False

    def _on_spectrum_changed(
        self,
        event: SpectrumAddedPayload | SpectrumRemovedPayload | SpectrumChangedPayload,
    ) -> None:
        self.cancel(event.name)
        self.cancel("selection")

    def _on_selection_changed(self, _event: SelectionChangedPayload) -> None:
        active = self._contexts.get(None)
        if active is not None and isinstance(active.context, CrossSelectionContext):
            if self._publishing_selection:
                active.context.selection_version = self._state.version.get(
                    SELECTION_VERSION_KEY
                )
            else:
                self.cancel("selection")

    def begin_line_pick(self, name: str) -> LinePickContext:
        """Reuse a named context or open captured-input device-axis picking.

        InvalidInputError rejects unknown names. FailedPreconditionError rejects
        a disposed owner. Seed inherits meaningful alignment;
        OneTone fixes magnitude_only to True. Reentrant retirement that changes
        context or spectrum/selection publications, or disposes the
        owner raises FailedPreconditionError and keeps the callback's result.
        RuntimeError rejects off-owner use.
        """
        self._require_owner()
        if self._disposed:
            raise FailedPreconditionError("interactive owner is disposed")
        if name not in self._state.spectrums:
            raise InvalidInputError(f"unknown spectrum {name!r}")
        checkpoint = self._capture_begin_checkpoint()
        current = self.current_line_pick(name)
        self._check_begin_checkpoint(checkpoint)
        if current is not None:
            return current
        self._cancel_for_begin(checkpoint, name)
        entry = self._state.spectrums[name]
        inputs = FluxPickInputs(
            entry.raw["signals"], entry.raw["dev_values"], entry.raw["freqs"]
        )
        seeded = entry.alignment_seeded or entry.aligned
        half, integer = fold_initial_lines(
            inputs.dev_values,
            entry.flux_half if seeded else None,
            entry.flux_int if seeded else None,
        )
        plugin = SharedFluxPickPlugin(
            inputs,
            FluxPickState(
                flux_half=half,
                flux_int=integer,
                magnitude_only=entry.spec_type == "OneTone",
            ),
            build_result=lambda state: analyze_flux_pick(inputs, state),
        )
        if self._background is not None:
            plugin.bind_background(self._background)
        context = LinePickContext(
            name, plugin, plugin.open(self._owner), inherited_from=entry.inherited_from
        )
        self._install_context(context)
        return context

    def current_line_pick(self, name: str | None = None) -> LinePickContext | None:
        """Read a named valid line draft; omitted name follows GUI display."""
        active = self.inspect(name)
        return (
            active.context
            if active is not None and isinstance(active.context, LinePickContext)
            else None
        )

    def finish_line_pick(
        self, name: str | None = None
    ) -> PublicationReceipt[FluxPickAnalysis]:
        """Validate, close and publish alignment; return accepted numeric result.

        name addresses a literal spectrum; omission reads the displayed draft.
        Missing context or invalid separation raises FailedPreconditionError;
        validation failure keeps input editable. Clear context before publication.
        Publication errors propagate and do not reopen terminal input.
        """
        self._require_owner()
        context = self.current_line_pick(name)
        if context is None:
            raise FailedPreconditionError("no active line picker")
        # Validate while editable; retire before publication, even if result building fails.
        context.plugin.can_finish(context.session.snapshot())
        try:
            result = context.plugin.finish(context.session)
        finally:
            self.cancel(context.spectrum_name)
        invalidated = self._publish_alignment(
            context.spectrum_name,
            result.flux_half,
            result.flux_int,
            None if context.alignment_changed else context.inherited_from,
        )
        return PublicationReceipt(result, invalidated)

    def begin_sample_pick(self, name: str) -> OneTonePickContext | TwoTonePickContext:
        """Open/reuse sampling for the named spectrum, using its published type.

        Unknown names raise InvalidInputError. The native picker rejects
        unaligned sources or a disposed owner with FailedPreconditionError.
        Return its actual OneTone/TwoTone context, retaining the same lifetime,
        replacement and owner-thread rules as the explicit picker entry points.
        """
        self._require_owner()
        entry = self._state.get_spectrum(name)
        if entry.spec_type == "OneTone":
            return self.begin_onetone_pick(name)
        return self.begin_twotone_pick(name)

    def begin_onetone_pick(self, name: str) -> OneTonePickContext:
        """Reuse a named context or open picking for an aligned OneTone.

        Unknown name raises InvalidInputError. Unaligned/wrong-type or
        disposed owner raises FailedPreconditionError. Switching picker kind
        closes old input before replacement. Reentrant retirement that changes
        context or spectrum/selection publications, or disposes the
        owner raises FailedPreconditionError and keeps the callback's result.
        Off-owner use raises RuntimeError.
        """
        self._require_owner()
        if self._disposed:
            raise FailedPreconditionError("interactive owner is disposed")
        if name not in self._state.spectrums:
            raise InvalidInputError(f"unknown spectrum {name!r}")
        entry = self._state.spectrums[name]
        if entry.spec_type != "OneTone":
            raise FailedPreconditionError(
                "one-tone picking requires a OneTone spectrum"
            )
        if not entry.aligned:
            raise FailedPreconditionError(
                "one-tone picking requires an aligned spectrum"
            )
        checkpoint = self._capture_begin_checkpoint()
        current = self.current_onetone_pick(name)
        self._check_begin_checkpoint(checkpoint)
        if current is not None:
            return current
        self._cancel_for_begin(checkpoint, name)
        inputs = OneToneInputs(
            FluxPickInputs(
                entry.raw["signals"], entry.raw["dev_values"], entry.raw["freqs"]
            )
        )
        plugin = OneTonePickPlugin(inputs)
        context = OneTonePickContext(
            name, plugin, plugin.open(self._owner), entry.flux_half, entry.flux_int
        )
        self._install_context(context)
        return context

    def current_onetone_pick(
        self, name: str | None = None
    ) -> OneTonePickContext | None:
        """Read a named valid onetone draft; omitted name follows GUI display."""
        active = self.inspect(name)
        return (
            active.context
            if active is not None and isinstance(active.context, OneTonePickContext)
            else None
        )

    def finish_onetone_pick(
        self, name: str | None = None
    ) -> PublicationReceipt[OneTonePickResult]:
        """Validate committed indices, close input and publish native point arrays.

        name addresses a literal spectrum; omission reads the displayed draft.
        Missing context raises FailedPreconditionError; invalid state retains
        editable input. Clear ownership before Controller's publication.
        Publication failure propagates without reopening terminal input.
        """
        self._require_owner()
        context = self.current_onetone_pick(name)
        if context is None:
            raise FailedPreconditionError("no active one-tone picker")
        context.plugin.can_finish(context.session.snapshot())
        try:
            result = context.plugin.finish(context.session)
        finally:
            self.cancel(context.spectrum_name)
        invalidated = self._publish_points(
            context.spectrum_name, result.dev_values, result.freqs
        )
        return PublicationReceipt(result, invalidated)

    def begin_twotone_pick(self, name: str) -> TwoTonePickContext:
        """Reuse a named aligned TwoTone context or open a captured seed.

        Unknown name raises InvalidInputError. Unaligned/wrong-type or
        disposed owner raises FailedPreconditionError. Kind switch closes old
        input. Reentrant retirement that changes context, spectrum/selection
        publications, or disposes the owner raises
        FailedPreconditionError and keeps the callback's result. Off-owner use
        raises RuntimeError. Invalid raw data raises ValueError.
        """
        self._require_owner()
        if self._disposed:
            raise FailedPreconditionError("interactive owner is disposed")
        if name not in self._state.spectrums:
            raise InvalidInputError(f"unknown spectrum {name!r}")
        entry = self._state.spectrums[name]
        if entry.spec_type != "TwoTone":
            raise FailedPreconditionError(
                "two-tone picking requires a TwoTone spectrum"
            )
        if not entry.aligned:
            raise FailedPreconditionError(
                "two-tone picking requires an aligned spectrum"
            )
        checkpoint = self._capture_begin_checkpoint()
        current = self.current_twotone_pick(name)
        self._check_begin_checkpoint(checkpoint)
        if current is not None:
            return current
        self._cancel_for_begin(checkpoint, name)
        inputs = TwoToneInputs(
            FluxPickInputs(
                entry.raw["signals"], entry.raw["dev_values"], entry.raw["freqs"]
            )
        )
        plugin = TwoTonePickPlugin(inputs)
        context = TwoTonePickContext(
            name, plugin, plugin.open(self._owner), entry.flux_half, entry.flux_int
        )
        self._install_context(context)
        return context

    def current_twotone_pick(
        self, name: str | None = None
    ) -> TwoTonePickContext | None:
        """Read a named valid twotone draft; omitted name follows GUI display."""
        active = self.inspect(name)
        return (
            active.context
            if active is not None and isinstance(active.context, TwoTonePickContext)
            else None
        )

    def finish_twotone_pick(
        self, name: str | None = None
    ) -> PublicationReceipt[TwoTonePickResult]:
        """Compute latest committed points, close input and publish through PointsService.

        name addresses a literal spectrum; omission reads the displayed draft.
        Missing context raises FailedPreconditionError. Validation failure keeps
        context editable. Clear context before publication; publication failure
        remains terminal. Never use a widget's pending or cached preview.
        """
        self._require_owner()
        context = self.current_twotone_pick(name)
        if context is None:
            raise FailedPreconditionError("no active two-tone picker")
        context.plugin.can_finish(context.session.snapshot())
        try:
            result = context.plugin.finish(context.session)
        finally:
            self.cancel(context.spectrum_name)
        invalidated = self._publish_points(
            context.spectrum_name, result.dev_values, result.freqs
        )
        return PublicationReceipt(result, invalidated)

    def begin_cross_selection(self) -> CrossSelectionContext:
        """Reuse valid open cross context or capture an all-selected cloud seed.

        Capture every source version, including zero-point entries, and bounds
        from the usable spectra's raw axes and points. Only published distance is
        inherited. Spectrum drafts remain live when Filter opens.
        FailedPreconditionError rejects disposed owner or no usable cloud;
        invalid numeric input raises ValueError. Reentrant retirement that changes
        context or spectrum/selection publications, or disposes the
        owner raises FailedPreconditionError and keeps the callback's result.
        Off-owner use raises RuntimeError.
        """
        self._require_owner()
        if self._disposed:
            raise FailedPreconditionError("interactive owner is disposed")
        checkpoint = self._capture_begin_checkpoint()
        current = self.current_cross_selection()
        self._check_begin_checkpoint(checkpoint)
        if current is not None:
            return current
        inputs = self._capture_cross_selection_inputs()
        plugin = CrossSelectionPlugin(
            inputs, min_distance=self._state.selection.min_distance
        )
        self._cancel_for_begin(checkpoint, "selection")
        context = CrossSelectionContext(
            plugin=plugin,
            session=plugin.open(self._owner),
            source_versions=tuple(
                (name, self._state.version.get(spectrum_version_key(name)))
                for name in self._state.spectrums
            ),
            source_point_counts=tuple(
                (name, entry.point_count)
                for name, entry in self._state.spectrums.items()
            ),
            spectrum_set_version=self._state.version.get(SPECTRUM_SET_VERSION_KEY),
            selection_version=self._state.version.get(SELECTION_VERSION_KEY),
        )
        self._install_context(context)
        return context

    def _capture_cross_selection_inputs(self) -> CrossSelectionInputs:
        fluxs, freqs = self._derive_pointcloud()
        if fluxs.size == 0:
            raise FailedPreconditionError(
                "cross-selection requires a usable point cloud"
            )
        backgrounds = tuple(
            CrossSelectionBackground(
                entry.name,
                entry.raw["fluxs"],
                entry.raw["freqs"],
                self._normalized_background(entry),
            )
            for entry in self._state.spectrums.values()
            if entry.point_count > 0
        )
        flux_bound = (
            min(float(fluxs.min()), *(float(bg.fluxs.min()) for bg in backgrounds)),
            max(float(fluxs.max()), *(float(bg.fluxs.max()) for bg in backgrounds)),
        )
        freq_bound = (
            min(float(freqs.min()), *(float(bg.freqs.min()) for bg in backgrounds)),
            max(float(freqs.max()), *(float(bg.freqs.max()) for bg in backgrounds)),
        )
        return CrossSelectionInputs(fluxs, freqs, backgrounds, flux_bound, freq_bound)

    def _normalized_background(self, entry: SpectrumEntry) -> NDArray[np.float64]:
        """Return entry's read-only normalized signals, reused while unchanged.

        A cached result is valid only for the same spectrum version and raw
        signals object; any republication recomputes it.
        """
        version = self._state.version.get(spectrum_version_key(entry.name))
        signals = entry.raw["signals"]
        cached = self._normalized_backgrounds.get(entry.name)
        if (
            cached is not None
            and cached.spectrum_version == version
            and cached.raw_signals is signals
        ):
            return cached.normalized_signals
        normalized = cast2real_and_norm(signals, use_phase=entry.spec_type == "TwoTone")
        normalized.flags.writeable = False
        self._normalized_backgrounds = {
            name: item
            for name, item in self._normalized_backgrounds.items()
            if name in self._state.spectrums
        }
        self._normalized_backgrounds[entry.name] = _NormalizedBackground(
            version, signals, normalized
        )
        return normalized

    def current_cross_selection(self) -> CrossSelectionContext | None:
        """Return valid open context or None on owner loop, without starting one.

        Verify collection version, all source names/versions and selection version.
        Invalid context closes input; same-length reload cannot reuse its cloud.
        Off-owner use raises RuntimeError. Reads never consume Undo.
        """
        self._require_owner()
        active = self._contexts.get(None)
        context = active.context if active is not None else None
        if not isinstance(context, CrossSelectionContext):
            return None
        source_versions = tuple(
            (name, self._state.version.get(spectrum_version_key(name)))
            for name in self._state.spectrums
        )
        if (
            self._disposed
            or context.source_versions != source_versions
            or context.spectrum_set_version
            != self._state.version.get(SPECTRUM_SET_VERSION_KEY)
            or context.selection_version
            != self._state.version.get(SELECTION_VERSION_KEY)
        ):
            self.cancel("selection")
            return None
        try:
            context.session.ensure_input_open()
        except FailedPreconditionError:
            self.cancel("selection")
            return None
        return context

    def apply_cross_selection(self) -> CrossSelectionResult:
        """Publish once without closing input; return detached native Filter numbers.

        Missing/invalid input raises FailedPreconditionError. Publication errors
        leave input editable without rollback. Same owner-only rules as finish.
        """
        return self._publish_cross_selection().result

    def _publish_cross_selection(self) -> PublicationReceipt[CrossSelectionResult]:
        """Synchronously publish the latest complete kept mask once, nonterminally.

        Validate current context and analyze its snapshot, never pending preview.
        Missing/invalid context raises FailedPreconditionError. Publish errors
        propagate and input/Undo remain editable; no rollback is claimed.
        Self-produced SelectionChanged updates the captured version rather than
        cancelling this context; external publication invalidates it.
        Release the self-publication guard in finally, also on failure.
        Off-owner use raises RuntimeError.
        """
        self._require_owner()
        context = self.current_cross_selection()
        if context is None:
            raise FailedPreconditionError("no valid cross-selection context")
        active = self._contexts.get(None)
        if active is None:
            raise FailedPreconditionError("cross-selection retired")
        result = analyze_cross_selection(
            context.plugin.inputs, context.session.snapshot()
        )
        self._publishing_selection = True
        try:
            # Publication and the detached caller result own independent masks.
            invalidated = self._publish_selection(
                result.selected.copy(), result.min_distance
            )
        finally:
            # A publisher may commit and then fail; preserve its actual version, not a rollback.
            context.selection_version = self._state.version.get(SELECTION_VERSION_KEY)
            self._publishing_selection = False
        self._updated(active)
        return PublicationReceipt(result, invalidated)

    def finish_cross_selection(self) -> PublicationReceipt[CrossSelectionResult]:
        """Publish the latest kept mask and retire Filter after successful publication.

        Require a valid global context. Publication failures propagate without
        closing input or claiming rollback. Off-owner use raises RuntimeError.
        """
        self._require_owner()
        context = self.current_cross_selection()
        result = self._publish_cross_selection()
        # Publication facts may retire this identity and open a successor.
        if self.current_cross_selection() is context:
            self.cancel("selection")
        return result

    def cancel(self, name: str | None = None) -> None:
        """Retire only the addressed draft; no publication or other draft loss."""
        self._require_owner()
        active = self._contexts.pop(self._target_key(name), None)
        if active is None:
            return
        if isinstance(active.context, CrossSelectionContext):
            self._view_selection = False
        active.context.session.close_input()
        for unsubscribe in self._context_subscriptions.pop(active.context_id, ()):
            unsubscribe()
        self._emit_interactive(active, "closed")

    def dispose(self) -> None:
        """Reject admission before retiring input, then release subscriptions.

        Idempotent owner-loop call. Synchronous closed callbacks cannot begin new
        input. EventBus logs and isolates ordinary callback exceptions. Source
        subscriptions are released even when retirement raises.
        """
        self._require_owner()
        if self._disposed:
            return
        self._disposed = True
        try:
            for key in tuple(self._contexts):
                self.cancel("selection" if key is None else key)
        finally:
            for unsubscribe in self._subscriptions:
                unsubscribe()
            self._subscriptions = ()
