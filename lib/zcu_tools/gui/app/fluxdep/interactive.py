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
    spectrum_version_key,
)
from zcu_tools.gui.app.fluxdep.twotone import TwoTonePickPlugin
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.interactive.flux_pick import SharedFluxPickPlugin
from zcu_tools.gui.interactive.plugin import BackgroundSubmitter
from zcu_tools.gui.interactive.session import Session
from zcu_tools.gui.session.ports import OwnerScheduler


@dataclass(frozen=True, slots=True)
class LinePickContext:
    """Live context owned by FluxDepInteractiveOwner, not by its widgets.

    spectrum_name identifies the active spectrum whose captured arrays are used.
    plugin supplies shared device-axis commands and the accepted alignment result.
    session is the sole committed state; old contexts retain readable snapshots
    after cancellation but reject writes.
    """

    spectrum_name: str
    plugin: SharedFluxPickPlugin[FluxPickAnalysis]
    session: Session[FluxPickState]


@dataclass(frozen=True, slots=True)
class OneTonePickContext:
    """Live one-tone context with captured calibration references.

    spectrum_name identifies the active aligned OneTone spectrum.
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

    spectrum_name identifies the active aligned TwoTone spectrum. plugin owns
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
    Views may detach; closed old sessions reject writes but retain snapshots.
    """

    plugin: CrossSelectionPlugin
    session: Session[CrossSelectionState]
    source_versions: tuple[tuple[str, int], ...]
    spectrum_set_version: int
    selection_version: int


InteractiveContext = (
    LinePickContext | OneTonePickContext | TwoTonePickContext | CrossSelectionContext
)


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
class FluxDepInteractivePorts:
    """Runtime collaborators for app-owned picking and joint-cloud selection.

    background delivers computation on the owner loop; None rejects auto-align.
    publish_alignment receives spectrum name and native device half/integer lines.
    publish_points receives name and native device/GHz arrays; PointsService owns
    sorting/calibration. derive_pointcloud queries the complete calibrated
    flux/GHz cloud in spectrum insertion order. publish_selection receives that
    cloud's full bool kept mask and normalized distance; Controller owns its
    single version bump/fact. Publication errors propagate without rollback.
    """

    background: BackgroundSubmitter | None
    publish_alignment: Callable[[str, float, float], None]
    publish_points: Callable[[str, NDArray[np.float64], NDArray[np.float64]], None]
    derive_pointcloud: Callable[[], tuple[NDArray[np.float64], NDArray[np.float64]]]
    publish_selection: Callable[[NDArray[np.bool_], float], None]


class FluxDepInteractiveOwner:
    """Serialize one active picker and invalidate it on spectrum facts.

    Public access requires the supplied owner thread. Views may detach without
    ending a context. Switch, reload, removal and external spectrum changes close
    old input; dispose also releases bus subscriptions.
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
        self._context: InteractiveContext | None = None
        self._context_id = 0
        self._next_context_id = 1
        self._context_subscriptions: tuple[Callable[[], None], ...] = ()
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

    def inspect(self) -> ActiveInteractiveContext | None:
        """Read the valid live context on the owner loop, without starting one.

        Return None when absent, disposed, closed, or invalidated by source facts.
        Source invalidation retires old input as the current_* queries do.
        Valid reads never publish, change active spectrum, or consume Undo.
        RuntimeError rejects access outside the supplied owner thread.
        """
        self._require_owner()
        context = self._context
        if isinstance(context, LinePickContext):
            context = self.current_line_pick()
        elif isinstance(context, OneTonePickContext):
            context = self.current_onetone_pick()
        elif isinstance(context, TwoTonePickContext):
            context = self.current_twotone_pick()
        elif isinstance(context, CrossSelectionContext):
            context = self.current_cross_selection()
        if context is None:
            return None
        try:
            context.session.ensure_input_open()
        except FailedPreconditionError:
            self.cancel()
            return None
        return ActiveInteractiveContext(self._context_id, context)

    def _install_context(self, context: InteractiveContext) -> None:
        self._context = context
        self._context_id = self._next_context_id
        self._next_context_id += 1
        active = ActiveInteractiveContext(self._context_id, context)
        subscriptions = [context.session.subscribe(lambda: self._updated(active))]
        if isinstance(context, LinePickContext):
            subscriptions.append(
                context.plugin.subscribe_alignment(
                    lambda _busy, _error: self._updated(active)
                )
            )
        self._context_subscriptions = tuple(subscriptions)
        self._emit_interactive(active, "opened")

    def _updated(self, active: ActiveInteractiveContext) -> None:
        if self._context is active.context and self._context_id == active.context_id:
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

    def _on_active_changed(self, event: ActiveSpectrumChangedPayload) -> None:
        if isinstance(self._context, CrossSelectionContext) or (
            self._context is not None and event.name != self._context.spectrum_name
        ):
            self.cancel()

    def _on_spectrum_changed(
        self,
        event: SpectrumAddedPayload | SpectrumRemovedPayload | SpectrumChangedPayload,
    ) -> None:
        if isinstance(self._context, CrossSelectionContext) or (
            self._context is not None and event.name == self._context.spectrum_name
        ):
            self.cancel()

    def _on_selection_changed(self, _event: SelectionChangedPayload) -> None:
        if isinstance(self._context, CrossSelectionContext):
            if self._publishing_selection:
                self._context.selection_version = self._state.version.get(
                    SELECTION_VERSION_KEY
                )
            else:
                self.cancel()

    def begin_line_pick(self, name: str) -> LinePickContext:
        """Reuse valid active context or open captured-input device-axis picking.

        InvalidInputError rejects unknown names. FailedPreconditionError rejects
        inactive names or a disposed owner. Seed inherits meaningful alignment;
        OneTone fixes magnitude_only to True. RuntimeError rejects off-owner use.
        """
        self._require_owner()
        if self._disposed:
            raise FailedPreconditionError("interactive owner is disposed")
        if name not in self._state.spectrums:
            raise InvalidInputError(f"unknown spectrum {name!r}")
        if name != self._state.active_spectrum:
            raise FailedPreconditionError("line picking requires the active spectrum")
        current = self.current_line_pick()
        if current is not None:
            return current
        self.cancel()
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
        context = LinePickContext(name, plugin, plugin.open(self._owner))
        self._install_context(context)
        return context

    def current_line_pick(self) -> LinePickContext | None:
        """Return the valid active context or None without creating a session."""
        self._require_owner()
        if (
            self._context is not None
            and not isinstance(self._context, CrossSelectionContext)
            and (
                self._context.spectrum_name != self._state.active_spectrum
                or self._context.spectrum_name not in self._state.spectrums
            )
        ):
            self.cancel()
        return self._context if isinstance(self._context, LinePickContext) else None

    def finish_line_pick(self) -> FluxPickAnalysis:
        """Validate, close and publish alignment; return accepted numeric result.

        Missing context or invalid separation raises FailedPreconditionError;
        validation failure keeps input editable. Clear context before publication.
        Publication errors propagate and do not reopen terminal input.
        """
        self._require_owner()
        context = self.current_line_pick()
        if context is None:
            raise FailedPreconditionError("no active line picker")
        # Validate while editable; retire before publication, even if result building fails.
        context.plugin.can_finish(context.session.snapshot())
        try:
            result = context.plugin.finish(context.session)
        finally:
            self.cancel()
        self._publish_alignment(
            context.spectrum_name, result.flux_half, result.flux_int
        )
        return result

    def begin_onetone_pick(self, name: str) -> OneTonePickContext:
        """Reuse a valid context or open picking for an active aligned OneTone.

        Unknown name raises InvalidInputError. Inactive/unaligned/wrong-type or
        disposed owner raises FailedPreconditionError. Switching picker kind
        closes old input before replacement. Off-owner use raises RuntimeError.
        """
        self._require_owner()
        if self._disposed:
            raise FailedPreconditionError("interactive owner is disposed")
        if name not in self._state.spectrums:
            raise InvalidInputError(f"unknown spectrum {name!r}")
        if name != self._state.active_spectrum:
            raise FailedPreconditionError(
                "one-tone picking requires the active spectrum"
            )
        entry = self._state.spectrums[name]
        if entry.spec_type != "OneTone":
            raise FailedPreconditionError(
                "one-tone picking requires a OneTone spectrum"
            )
        if not entry.aligned:
            raise FailedPreconditionError(
                "one-tone picking requires an aligned spectrum"
            )
        current = self.current_onetone_pick()
        if current is not None:
            return current
        self.cancel()
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

    def current_onetone_pick(self) -> OneTonePickContext | None:
        """Read the valid OneTone context, or None, without creating a session.

        Requires owner loop; absent, invalidated or another picker kind returns
        None. Invalid spectrum identity closes old input as for line picking.
        """
        self._require_owner()
        if (
            self._context is not None
            and not isinstance(self._context, CrossSelectionContext)
            and (
                self._context.spectrum_name != self._state.active_spectrum
                or self._context.spectrum_name not in self._state.spectrums
            )
        ):
            self.cancel()
        if not isinstance(self._context, OneTonePickContext):
            return None
        try:
            self._context.session.ensure_input_open()
        except FailedPreconditionError:
            self.cancel()
            return None
        return self._context

    def finish_onetone_pick(self) -> OneTonePickResult:
        """Validate committed indices, close input and publish native point arrays.

        Missing context raises FailedPreconditionError; invalid state retains
        editable input. Clear ownership before Controller's publication.
        Publication failure propagates without reopening terminal input.
        """
        self._require_owner()
        context = self.current_onetone_pick()
        if context is None:
            raise FailedPreconditionError("no active one-tone picker")
        context.plugin.can_finish(context.session.snapshot())
        try:
            result = context.plugin.finish(context.session)
        finally:
            self.cancel()
        self._publish_points(context.spectrum_name, result.dev_values, result.freqs)
        return result

    def begin_twotone_pick(self, name: str) -> TwoTonePickContext:
        """Reuse valid active aligned TwoTone context or open a captured seed.

        Unknown name raises InvalidInputError. Inactive/unaligned/wrong-type or
        disposed owner raises FailedPreconditionError. Kind switch closes old
        input. Off-owner use raises RuntimeError. Invalid raw data raises ValueError.
        """
        self._require_owner()
        if self._disposed:
            raise FailedPreconditionError("interactive owner is disposed")
        if name not in self._state.spectrums:
            raise InvalidInputError(f"unknown spectrum {name!r}")
        if name != self._state.active_spectrum:
            raise FailedPreconditionError(
                "two-tone picking requires the active spectrum"
            )
        entry = self._state.spectrums[name]
        if entry.spec_type != "TwoTone":
            raise FailedPreconditionError(
                "two-tone picking requires a TwoTone spectrum"
            )
        if not entry.aligned:
            raise FailedPreconditionError(
                "two-tone picking requires an aligned spectrum"
            )
        current = self.current_twotone_pick()
        if current is not None:
            return current
        self.cancel()
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

    def current_twotone_pick(self) -> TwoTonePickContext | None:
        """Return valid open TwoTone context, otherwise None, on the owner loop.

        Invalid active/spectrum identity closes the old input, as for OneTone.
        This query does not create a Session or compute a preview.
        """
        self._require_owner()
        if (
            self._context is not None
            and not isinstance(self._context, CrossSelectionContext)
            and (
                self._context.spectrum_name != self._state.active_spectrum
                or self._context.spectrum_name not in self._state.spectrums
            )
        ):
            self.cancel()
        if not isinstance(self._context, TwoTonePickContext):
            return None
        entry = self._state.spectrums[self._context.spectrum_name]
        if entry.spec_type != "TwoTone" or not entry.aligned:
            self.cancel()
            return None
        try:
            self._context.session.ensure_input_open()
        except FailedPreconditionError:
            self.cancel()
            return None
        return self._context

    def finish_twotone_pick(self) -> TwoTonePickResult:
        """Compute latest committed points, close input and publish through PointsService.

        Missing context raises FailedPreconditionError. Validation failure keeps
        context editable. Clear context before publication; publication failure
        remains terminal. Never use a widget's pending or cached preview.
        """
        self._require_owner()
        context = self.current_twotone_pick()
        if context is None:
            raise FailedPreconditionError("no active two-tone picker")
        context.plugin.can_finish(context.session.snapshot())
        try:
            result = context.plugin.finish(context.session)
        finally:
            self.cancel()
        self._publish_points(context.spectrum_name, result.dev_values, result.freqs)
        return result

    def begin_cross_selection(self) -> CrossSelectionContext:
        """Reuse valid open cross context or capture an all-selected cloud seed.

        Capture every source version, including zero-point entries, and bounds
        from the usable spectra's raw axes and points. Only published distance is
        inherited. Cancel the previous picker on successful replacement.
        FailedPreconditionError rejects disposed owner or no usable cloud;
        invalid numeric input raises ValueError. Off-owner use raises RuntimeError.
        """
        self._require_owner()
        if self._disposed:
            raise FailedPreconditionError("interactive owner is disposed")
        current = self.current_cross_selection()
        if current is not None:
            return current
        inputs = self._capture_cross_selection_inputs()
        plugin = CrossSelectionPlugin(
            inputs, min_distance=self._state.selection.min_distance
        )
        self.cancel()
        context = CrossSelectionContext(
            plugin=plugin,
            session=plugin.open(self._owner),
            source_versions=tuple(
                (name, self._state.version.get(spectrum_version_key(name)))
                for name in self._state.spectrums
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
                cast2real_and_norm(
                    entry.raw["signals"], use_phase=entry.spec_type == "TwoTone"
                ),
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

    def current_cross_selection(self) -> CrossSelectionContext | None:
        """Return valid open context or None on owner loop, without starting one.

        Verify collection version, all source names/versions and selection version.
        Invalid context closes input; same-length reload cannot reuse its cloud.
        Off-owner use raises RuntimeError. Reads never consume Undo.
        """
        self._require_owner()
        context = self._context
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
            self.cancel()
            return None
        try:
            context.session.ensure_input_open()
        except FailedPreconditionError:
            self.cancel()
            return None
        return context

    def apply_cross_selection(self) -> CrossSelectionResult:
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
        active = ActiveInteractiveContext(self._context_id, context)
        result = analyze_cross_selection(
            context.plugin.inputs, context.session.snapshot()
        )
        self._publishing_selection = True
        try:
            # Publication and the detached caller result own independent masks.
            self._publish_selection(result.selected.copy(), result.min_distance)
        finally:
            # A publisher may commit and then fail; preserve its actual version, not a rollback.
            context.selection_version = self._state.version.get(SELECTION_VERSION_KEY)
            self._publishing_selection = False
        self._updated(active)
        return result

    def cancel(self) -> None:
        """Close current input without publication; safe with no active context."""
        self._require_owner()
        context = self._context
        if context is None:
            return
        active = ActiveInteractiveContext(self._context_id, context)
        context.session.close_input()
        self._context = None
        self._context_id = 0
        for unsubscribe in self._context_subscriptions:
            unsubscribe()
        self._context_subscriptions = ()
        self._emit_interactive(active, "closed")

    def dispose(self) -> None:
        """Idempotently cancel and release subscriptions; reject future begin."""
        self._require_owner()
        if self._disposed:
            return
        self.cancel()
        self._disposed = True
        for unsubscribe in self._subscriptions:
            unsubscribe()
        self._subscriptions = ()
