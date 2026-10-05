"""Shared device-axis line picking for measure and fluxdep owners."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

from zcu_tools.analysis.fluxdep.line_state import (
    FluxLineRole,
    FluxPickInputs,
    FluxPickState,
    align_lines,
    move_line,
    swap_lines,
)
from zcu_tools.analysis.fluxdep.processing import cast2real_and_norm
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.interactive.plugin import Action, Command, PluginDefinition
from zcu_tools.gui.interactive.session import Session
from zcu_tools.gui.remote.errors import RemoteError
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec, validate_params

R = TypeVar("R")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FluxPickActions:
    """Actions against the latest committed device-axis state.

    move sets a half/integer role to a native device position.
    conjugate enables translating both lines on subsequent moves.
    swap exchanges the two roles. apply_alignment installs a calculated pair.
    Invalid roles, nonfinite positions or insufficient separation reject commits.
    """

    move: Action[FluxPickState, tuple[FluxLineRole, float]]
    conjugate: Action[FluxPickState, bool]
    swap: Action[FluxPickState, None]
    apply_alignment: Action[FluxPickState, tuple[float, float]]


class SharedFluxPickPlugin(PluginDefinition[FluxPickState, R], Generic[R]):
    """Qt-free actions, command validation and single-flight auto alignment.

    Inputs own read-only arrays. Each instance binds one operation's background
    runner; all command/status access belongs to its Session owner loop.
    Terminal sessions reject input and ignore late alignment publication.
    """

    inputs: FluxPickInputs
    actions: FluxPickActions

    def __init__(
        self,
        inputs: FluxPickInputs,
        seed: FluxPickState,
        *,
        build_result: Callable[[FluxPickState], R],
    ) -> None:
        """Capture inputs and seed; validate line separation before terminal build.

        build_result converts accepted state to the app result after input closes.
        Its failure leaves input closed. Commands are move_line(role, position),
        set_conjugate(enabled), swap_lines() and auto_align().
        """
        move_params = (
            ParamSpec("role", JsonType.STRING, enum=("half", "integer")),
            ParamSpec("position", JsonType.NUMBER),
        )
        conjugate_params = (ParamSpec("enabled", JsonType.BOOLEAN),)
        actions = FluxPickActions(
            move=Action(lambda state, pair: _move(state, pair, inputs.min_distance)),
            conjugate=Action(_set_conjugate),
            swap=Action(lambda state, _unused: swap_lines(state)),
            apply_alignment=Action(
                lambda state, pair: _apply_alignment(state, pair, inputs.min_distance)
            ),
        )

        def move_command(
            session: Session[FluxPickState], params: Mapping[str, object]
        ) -> FluxPickState:
            values = _validate_command(move_params, params)
            role = values["role"]
            position = values["position"]
            if role not in ("half", "integer") or not isinstance(position, float):
                raise InvalidInputError(
                    "move_line requires a role and numeric position"
                )
            # Explicit branches narrow the wire string to the domain role.
            pair: tuple[FluxLineRole, float] = (
                "half" if role == "half" else "integer",
                position,
            )
            return actions.move.execute(session, pair)

        def conjugate_command(
            session: Session[FluxPickState], params: Mapping[str, object]
        ) -> FluxPickState:
            enabled = _validate_command(conjugate_params, params)["enabled"]
            if not isinstance(enabled, bool):
                raise InvalidInputError("conjugate requires a boolean")
            return actions.conjugate.execute(session, enabled)

        PluginDefinition.__init__(
            self,
            plugin_id="flux_pick",
            seed=seed,
            commands=(
                Command("move_line", move_params, move_command),
                Command("set_conjugate", conjugate_params, conjugate_command),
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
            can_finish=lambda state: _require_finishable(state, inputs.min_distance),
            build_result=build_result,
        )
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "_alignment_busy", False)
        object.__setattr__(self, "_alignment_error", None)
        object.__setattr__(self, "_alignment_listeners", {})
        object.__setattr__(self, "_next_listener", 0)

    _alignment_busy: bool
    _alignment_error: str | None
    _alignment_listeners: dict[int, Callable[[bool, str | None], None]]
    _next_listener: int

    @property
    def alignment_busy(self) -> bool:
        """Whether this plugin has an unsettled alignment worker."""
        return self._alignment_busy

    def info(self) -> Mapping[str, object]:
        """Return alignment_busy and nullable alignment_error, separate from state."""
        return {
            "alignment_busy": self._alignment_busy,
            "alignment_error": self._alignment_error,
        }

    def subscribe_alignment(
        self, callback: Callable[[bool, str | None], None]
    ) -> Callable[[], None]:
        """Observe owner-loop busy/error changes; return idempotent cleanup."""
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
        """Return captured state after starting one worker, without a state commit.

        FailedPreconditionError rejects terminal input, an unbound runner or a
        second worker. Submission errors propagate after clearing busy status.
        Completion commits against the latest state on the owner loop.
        """
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

        def compute() -> FluxPickState:
            half, integer = self.calculate_alignment(captured)
            # A nominal carrier preserves types across the runner's object boundary.
            return replace(captured, flux_half=half, flux_int=integer)

        def on_done(value: object) -> None:
            try:
                session.ensure_input_open()
            except FailedPreconditionError:
                settle()
                return
            try:
                if not isinstance(value, FluxPickState):
                    raise TypeError("alignment worker must return FluxPickState")
                self.actions.apply_alignment.execute(
                    session, (value.flux_half, value.flux_int)
                )
            except Exception as exc:
                logger.exception("interactive alignment delivery failed")
                settle(exc)
            else:
                settle()

        try:
            self.run_background(compute, on_done, settle)
        except Exception as exc:
            settle(exc)
            raise
        return captured

    def calculate_alignment(self, state: FluxPickState) -> tuple[float, float]:
        """Compute half/integer device positions from captured inputs, without writes."""
        projection = cast2real_and_norm(
            self.inputs.signals, use_phase=not state.magnitude_only
        )
        aligned = align_lines(state, self.inputs.dev_values, projection)
        return aligned.flux_half, aligned.flux_int


def _validate_command(
    specs: tuple[ParamSpec, ...], params: Mapping[str, object]
) -> Mapping[str, object]:
    try:
        return validate_params(specs, params)
    except RemoteError as exc:
        raise InvalidInputError(str(exc)) from exc


def _move(
    state: FluxPickState, pair: tuple[FluxLineRole, float], min_distance: float
) -> FluxPickState:
    if isinstance(pair[1], bool):
        raise InvalidInputError("position must be a number, not a boolean")
    try:
        candidate = move_line(state, pair[0], pair[1], min_distance=min_distance)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    if abs(candidate.flux_int - candidate.flux_half) < min_distance:
        raise InvalidInputError("flux lines must remain separated")
    return candidate


def _require_finishable(state: FluxPickState, min_distance: float) -> None:
    if abs(state.flux_int - state.flux_half) < min_distance:
        raise FailedPreconditionError("flux lines must remain separated")


def _set_conjugate(state: FluxPickState, enabled: object) -> FluxPickState:
    if not isinstance(enabled, bool):
        raise InvalidInputError("conjugate requires a boolean")
    return replace(state, conjugate=enabled)


def _apply_alignment(
    state: FluxPickState, positions: tuple[float, float], min_distance: float
) -> FluxPickState:
    if any(isinstance(position, bool) for position in positions):
        raise InvalidInputError("positions must be numbers, not booleans")
    try:
        candidate = replace(state, flux_half=positions[0], flux_int=positions[1])
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    if abs(candidate.flux_int - candidate.flux_half) < min_distance:
        raise InvalidInputError("flux lines must remain separated")
    return candidate
