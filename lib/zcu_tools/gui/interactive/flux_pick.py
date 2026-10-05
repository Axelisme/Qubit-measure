"""Shared device-axis line picking for measure and fluxdep owners."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from zcu_tools.analysis.fluxdep.line_state import (
    FluxLineRole,
    FluxPickInputs,
    FluxPickState,
)
from zcu_tools.gui.interactive.plugin import Action, PluginDefinition
from zcu_tools.gui.interactive.session import Session

R = TypeVar("R")


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
        raise NotImplementedError

    @property
    def alignment_busy(self) -> bool:
        """Whether this plugin has an unsettled alignment worker."""
        raise NotImplementedError

    def info(self) -> Mapping[str, object]:
        """Return alignment_busy and nullable alignment_error, separate from state."""
        raise NotImplementedError

    def subscribe_alignment(
        self, callback: Callable[[bool, str | None], None]
    ) -> Callable[[], None]:
        """Observe owner-loop busy/error changes; return idempotent cleanup."""
        raise NotImplementedError

    def start_alignment(self, session: Session[FluxPickState]) -> FluxPickState:
        """Return captured state after starting one worker, without a state commit.

        FailedPreconditionError rejects terminal input, an unbound runner or a
        second worker. Submission errors propagate after clearing busy status.
        Completion commits against the latest state on the owner loop.
        """
        raise NotImplementedError

    def calculate_alignment(self, state: FluxPickState) -> tuple[float, float]:
        """Compute half/integer device positions from captured inputs, without writes."""
        raise NotImplementedError
