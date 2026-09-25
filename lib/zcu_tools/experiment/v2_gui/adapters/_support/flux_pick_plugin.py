"""Shared flux-pick actions and wire commands for both measure adapters."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

from zcu_tools.analysis.fluxdep.line_state import (
    FluxLineRole,
    FluxPickInputs,
    FluxPickState,
    align_lines,
    fold_initial_lines,
    move_line,
    swap_lines,
)
from zcu_tools.analysis.fluxdep.processing import cast2real_and_norm
from zcu_tools.gui.app.main.adapter import AnalyzeRequest
from zcu_tools.gui.app.main.interactive import (
    Action,
    Command,
    PluginDefinition,
    Session,
)
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from .interactive_flux_pick import FluxPickResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FluxPickActions:
    move: Action[FluxPickState, tuple[FluxLineRole, float]]
    conjugate: Action[FluxPickState, bool]
    swap: Action[FluxPickState, None]
    apply_alignment: Action[FluxPickState, tuple[float, float]]


class FluxPickPlugin(PluginDefinition[FluxPickState, FluxPickResult]):
    """Captured numeric inputs and typed actions stay with this one operation."""

    __slots__ = (
        "inputs",
        "actions",
        "_alignment_busy",
        "_alignment_error",
        "_alignment_listeners",
        "_next_listener",
    )

    def __init__(self, inputs: FluxPickInputs, seed: FluxPickState) -> None:
        actions = FluxPickActions(
            move=Action(lambda state, pair: _move(state, pair, inputs.min_distance)),
            conjugate=Action(_set_conjugate),
            swap=Action(lambda state, _unused: swap_lines(state)),
            apply_alignment=Action(_apply_alignment),
        )
        PluginDefinition.__init__(
            self,
            plugin_id="flux_pick",
            seed=seed,
            commands=(
                Command[FluxPickState](
                    "move_line",
                    (
                        ParamSpec("role", JsonType.STRING),
                        ParamSpec("position", JsonType.NUMBER),
                    ),
                    lambda session, params: actions.move.execute(
                        session,
                        (
                            cast(FluxLineRole, params["role"]),
                            cast(float, params["position"]),
                        ),
                    ),
                ),
                Command[FluxPickState](
                    "set_conjugate",
                    (ParamSpec("enabled", JsonType.BOOLEAN),),
                    lambda session, params: actions.conjugate.execute(
                        session, cast(bool, params["enabled"])
                    ),
                ),
                Command[FluxPickState](
                    "swap_lines",
                    (),
                    lambda session, _params: actions.swap.execute(session, None),
                ),
                Command[FluxPickState](
                    "auto_align",
                    (),
                    lambda session, _params: self.start_alignment(session),
                ),
            ),
            can_finish=lambda _state: None,
            build_result=lambda state: FluxPickResult(
                flx_half=state.flux_half,
                flx_int=state.flux_int,
                flx_period=2 * abs(state.flux_int - state.flux_half),
            ),
            attach_figure=lambda result, figure: replace(result, figure=figure),
            project_state=lambda state: {
                "flux_half": state.flux_half,
                "flux_int": state.flux_int,
                "conjugate": state.conjugate,
                "magnitude_only": state.magnitude_only,
            },
        )
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "_alignment_busy", False)
        object.__setattr__(self, "_alignment_error", None)
        object.__setattr__(self, "_alignment_listeners", {})
        object.__setattr__(self, "_next_listener", 0)

    @property
    def alignment_busy(self) -> bool:
        return self._alignment_busy

    def info(self) -> Mapping[str, object]:
        return {
            "alignment_busy": self._alignment_busy,
            "alignment_error": self._alignment_error,
        }

    def subscribe_alignment(
        self, callback: Callable[[bool, str | None], None]
    ) -> Callable[[], None]:
        """Observe worker status without publishing a second committed state."""
        key = self._next_listener
        object.__setattr__(self, "_next_listener", key + 1)
        self._alignment_listeners[key] = callback

        def unsubscribe() -> None:
            self._alignment_listeners.pop(key, None)

        return unsubscribe

    def _notify_alignment(self) -> None:
        for callback in tuple(self._alignment_listeners.values()):
            try:
                callback(self._alignment_busy, self._alignment_error)
            except Exception:
                logger.exception("interactive alignment status subscriber failed")

    def start_alignment(self, session: Session[FluxPickState]) -> FluxPickState:
        """One worker for GUI and remote; its owner callback alone commits."""
        session.ensure_input_open()
        if self._alignment_busy:
            raise FailedPreconditionError("Auto Align is already running")
        captured = session.snapshot()
        object.__setattr__(self, "_alignment_busy", True)
        object.__setattr__(self, "_alignment_error", None)
        self._notify_alignment()

        def settle(error: Exception | None = None) -> None:
            object.__setattr__(self, "_alignment_busy", False)
            object.__setattr__(self, "_alignment_error", str(error) if error else None)
            if error is not None:
                logger.warning("interactive alignment failed: %s", error)
            self._notify_alignment()

        def on_done(value: object) -> None:
            try:
                session.ensure_input_open()
            except FailedPreconditionError:
                settle()
                return
            try:
                self.actions.apply_alignment.execute(
                    session, cast(tuple[float, float], value)
                )
            except Exception as exc:  # noqa: BLE001 - keep input editable
                settle(exc)
            else:
                settle()

        try:
            self.run_background(
                lambda: self.calculate_alignment(captured), on_done, settle
            )
        except Exception as exc:
            settle(exc)
            raise
        return captured

    def calculate_alignment(self, state: FluxPickState) -> tuple[float, float]:
        projection = cast2real_and_norm(
            self.inputs.signals, use_phase=not state.magnitude_only
        )
        aligned = align_lines(state, self.inputs.dev_values, projection)
        return aligned.flux_half, aligned.flux_int


def _move(
    state: FluxPickState, pair: tuple[FluxLineRole, float], min_distance: float
) -> FluxPickState:
    try:
        return move_line(state, pair[0], pair[1], min_distance=min_distance)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc


def _set_conjugate(state: FluxPickState, enabled: bool) -> FluxPickState:  # noqa: FBT001 - Action payload
    if type(enabled) is not bool:
        raise ValueError("conjugate requires a boolean")
    return replace(state, conjugate=enabled)


def _apply_alignment(
    state: FluxPickState, positions: tuple[float, float]
) -> FluxPickState:
    # FluxPickState validates both positions before Session publishes the replacement.
    return replace(state, flux_half=positions[0], flux_int=positions[1])


def make_flux_pick_plugin(
    req: AnalyzeRequest[Any, Any], *, force_magnitude: bool
) -> FluxPickPlugin:
    """Capture a read-only spectrum, fold the two calibrated seed positions."""
    result = req.run_result
    inputs = FluxPickInputs(result.signals, result.values, result.freqs)
    half, integer = fold_initial_lines(
        inputs.dev_values, req.md.get("flx_half", None), req.md.get("flx_int", None)
    )
    return FluxPickPlugin(
        inputs,
        FluxPickState(flux_half=half, flux_int=integer, magnitude_only=force_magnitude),
    )
