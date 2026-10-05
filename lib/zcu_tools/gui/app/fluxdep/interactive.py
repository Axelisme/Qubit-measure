"""Fluxdep-owned interactive contexts, independent of Qt presentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from zcu_tools.analysis.fluxdep.line_state import (
    FluxPickAnalysis,
    FluxPickInputs,
    FluxPickState,
    analyze_flux_pick,
    fold_initial_lines,
)
from zcu_tools.gui.app.fluxdep.event_bus import (
    ActiveSpectrumChangedPayload,
    EventBus,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.gui.app.fluxdep.state import FluxDepState
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
        background: BackgroundSubmitter | None,
        publish_alignment: Callable[[str, float, float], None],
    ) -> None:
        """Capture state and runtime ports; publish through Controller's alignment.

        background must deliver callbacks on owner, or None to reject auto_align.
        publish_alignment receives name and accepted native device positions.
        """
        self._state = state
        self._owner = owner
        self._require_owner()
        self._background = background
        self._publish_alignment = publish_alignment
        self._context: LinePickContext | None = None
        self._disposed = False
        self._subscriptions: tuple[Callable[[], None], ...] = (
            bus.subscribe(
                ActiveSpectrumChangedPayload, self._on_active_changed
            ).unsubscribe,
            bus.subscribe(SpectrumAddedPayload, self._on_spectrum_changed).unsubscribe,
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

    def _on_active_changed(self, event: ActiveSpectrumChangedPayload) -> None:
        if self._context is not None and event.name != self._context.spectrum_name:
            self.cancel()

    def _on_spectrum_changed(
        self,
        event: SpectrumAddedPayload | SpectrumRemovedPayload | SpectrumChangedPayload,
    ) -> None:
        if self._context is not None and event.name == self._context.spectrum_name:
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
        self._context = LinePickContext(name, plugin, plugin.open(self._owner))
        return self._context

    def current_line_pick(self) -> LinePickContext | None:
        """Return the valid active context or None without creating a session."""
        self._require_owner()
        if self._context is not None and (
            self._context.spectrum_name != self._state.active_spectrum
            or self._context.spectrum_name not in self._state.spectrums
        ):
            self.cancel()
        return self._context

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
        # Validate while editable; clear ownership before terminal result/publication.
        context.plugin.can_finish(context.session.snapshot())
        self._context = None
        result = context.plugin.finish(context.session)
        self._publish_alignment(
            context.spectrum_name, result.flux_half, result.flux_int
        )
        return result

    def cancel(self) -> None:
        """Close current input without publication; safe with no active context."""
        self._require_owner()
        if self._context is not None:
            self._context.session.close_input()
            self._context = None

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
