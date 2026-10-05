"""Fluxdep one-tone commands against a captured-input interactive session."""

from __future__ import annotations

from collections.abc import Mapping

from zcu_tools.analysis.fluxdep.onetone import (
    OneToneInputs,
    OneTonePickResult,
    OneTonePickState,
    analyze_onetone_pick,
    pick_onetone_state,
)
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.interactive.plugin import Action, Command, PluginDefinition
from zcu_tools.gui.interactive.session import Session
from zcu_tools.gui.remote.errors import RemoteError
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec, validate_params


class OneTonePickPlugin(PluginDefinition[OneTonePickState, OneTonePickResult]):
    """Share threshold commits between typed controls and command callers.

    inputs contains read-only captured spectrum and preprocessing.
    set_threshold computes a complete state at finite prominence [0, 5],
    translating ValueError to InvalidInputError without committing on failure.
    Inherited Session input closure rejects all terminal mutations.
    """

    inputs: OneToneInputs
    set_threshold: Action[OneTonePickState, float]

    def __init__(self, inputs: OneToneInputs, threshold: float = 1.0) -> None:
        """Seed the selection and declare set_threshold(threshold: NUMBER).

        Invalid seed raises ValueError. Commands reject invalid, boolean,
        unknown/missing, nonfinite or out-of-range values with InvalidInputError.
        Finish builds native device/GHz points, including an empty selection.
        """
        seed = pick_onetone_state(inputs, threshold)
        threshold_params = (ParamSpec("threshold", JsonType.NUMBER),)

        def calculate(_state: OneTonePickState, value: float) -> OneTonePickState:
            try:
                return pick_onetone_state(inputs, value)
            except ValueError as exc:
                raise InvalidInputError(str(exc)) from exc

        action = Action(calculate)

        def threshold_command(
            session: Session[OneTonePickState], params: Mapping[str, object]
        ) -> OneTonePickState:
            unknown = params.keys() - {"threshold"}
            if unknown:
                raise InvalidInputError(f"unknown parameters: {sorted(unknown)!r}")
            try:
                value = validate_params(threshold_params, params)["threshold"]
            except RemoteError as exc:
                raise InvalidInputError(str(exc)) from exc
            if not isinstance(value, float):
                raise InvalidInputError("threshold must be a number")
            return action.execute(session, value)

        def can_finish(state: OneTonePickState) -> None:
            # Validate through the numerical owner before Session closes input.
            analyze_onetone_pick(inputs, state)

        PluginDefinition.__init__(
            self,
            plugin_id="onetone_pick",
            seed=seed,
            commands=(Command("set_threshold", threshold_params, threshold_command),),
            can_finish=can_finish,
            build_result=lambda state: analyze_onetone_pick(inputs, state),
        )
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "set_threshold", action)
