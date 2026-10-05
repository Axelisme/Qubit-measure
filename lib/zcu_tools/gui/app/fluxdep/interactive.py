"""Fluxdep-owned interactive contexts, independent of Qt presentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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
from zcu_tools.gui.app.fluxdep.event_bus import (
    ActiveSpectrumChangedPayload,
    EventBus,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.gui.app.fluxdep.onetone import OneTonePickPlugin
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
        publish_points: Callable[[str, NDArray[np.float64], NDArray[np.float64]], None],
    ) -> None:
        """Capture state and runtime ports; publish through Controller's alignment.

        background must deliver callbacks on owner, or None to reject auto_align.
        publish_alignment receives name and accepted native device positions.
        publish_points receives name and uncalibrated device/GHz point arrays;
        Controller/PointsService owns sorting, calibration and publication.
        """
        self._state = state
        self._owner = owner
        self._require_owner()
        self._background = background
        self._publish_alignment = publish_alignment
        self._publish_points = publish_points
        self._context: LinePickContext | OneTonePickContext | None = None
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
        # Validate while editable; clear ownership before terminal result/publication.
        context.plugin.can_finish(context.session.snapshot())
        self._context = None
        result = context.plugin.finish(context.session)
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
        self._context = OneTonePickContext(
            name, plugin, plugin.open(self._owner), entry.flux_half, entry.flux_int
        )
        return self._context

    def current_onetone_pick(self) -> OneTonePickContext | None:
        """Read the valid OneTone context, or None, without creating a session.

        Requires owner loop; absent, invalidated or another picker kind returns
        None. Invalid spectrum identity closes old input as for line picking.
        """
        self._require_owner()
        if self._context is not None and (
            self._context.spectrum_name != self._state.active_spectrum
            or self._context.spectrum_name not in self._state.spectrums
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
        self._context = None
        result = context.plugin.finish(context.session)
        self._publish_points(context.spectrum_name, result.dev_values, result.freqs)
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
