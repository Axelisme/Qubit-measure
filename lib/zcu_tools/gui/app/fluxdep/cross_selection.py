"""Joint-cloud actions shared by GUI and command adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from zcu_tools.analysis.fluxdep.cross_selection import (
    CrossSelectionInputs,
    CrossSelectionResult,
    CrossSelectionState,
    analyze_cross_selection,
    fill_cross_selection_state,
    make_cross_selection_state,
    set_cross_selection_distance,
    set_cross_selection_tool,
    stroke_cross_selection_state,
)
from zcu_tools.analysis.fluxdep.stroke import BrushStroke, BrushTool
from zcu_tools.gui.app.fluxdep.brush_commands import (
    STROKE_PARAMS,
    TOOL_PARAMS,
    decode_brush_stroke,
    decode_brush_tool,
    decode_parameters,
)
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.interactive import Action, Command, PluginDefinition, Session
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec


class CrossSelectionPlugin(PluginDefinition[CrossSelectionState, CrossSelectionResult]):
    """Joint-cloud commands/actions over one captured input and Session.

    inputs owns read-only calibrated-flux/GHz numerical data. set_min_distance,
    stroke, perform_on_all and clear establish single Undo; set_tool preserves it.
    Apply is an app-owner command, not this definition's terminal finish.
    ValueError from numeric transitions becomes InvalidInputError without commit.
    Session owns off-owner/input-closed gates. Each action's payload is typed.
    """

    inputs: CrossSelectionInputs
    set_min_distance: Action[CrossSelectionState, float]
    set_tool: Action[CrossSelectionState, BrushTool]
    stroke: Action[CrossSelectionState, BrushStroke]
    perform_on_all: Action[CrossSelectionState, None]
    clear: Action[CrossSelectionState, None]

    def __init__(
        self,
        inputs: CrossSelectionInputs,
        *,
        min_distance: float = 0.0,
        brush_width: float = 0.05,
    ) -> None:
        """Declare cross_selection commands with an all-selected seed.

        min_distance/brush_width are finite normalized [0,0.1], width is radius.
        Invalid seeds raise ValueError. set_min_distance requires a NUMBER.
        set_tool accepts optional width/mode with at least one non-null field.
        stroke requires NUMBER_PAIRS vertices, NUMBER width and select/erase mode.
        perform_on_all and clear take no fields. Unknown/missing/wrong-type/domain
        values raise InvalidInputError atomically. GUI uses the same typed Actions.
        Owner Apply synchronously analyzes the snapshot without calling finish.
        """
        seed = make_cross_selection_state(
            inputs, min_distance=min_distance, width=brush_width
        )
        distance_params = (ParamSpec("min_distance", JsonType.NUMBER),)
        distance: Action[CrossSelectionState, float] = Action(
            lambda state, value: _transition(
                lambda: set_cross_selection_distance(inputs, state, value)
            )
        )
        tool: Action[CrossSelectionState, BrushTool] = Action(
            lambda state, value: _transition(
                lambda: set_cross_selection_tool(inputs, state, value)
            ),
            record_undo=False,
        )
        stroke: Action[CrossSelectionState, BrushStroke] = Action(
            lambda state, value: _transition(
                lambda: stroke_cross_selection_state(inputs, state, value)
            )
        )
        perform: Action[CrossSelectionState, None] = Action(
            lambda state, _: _transition(
                lambda: fill_cross_selection_state(
                    inputs, state, select=state.mode == "select"
                )
            )
        )
        clear: Action[CrossSelectionState, None] = Action(
            lambda state, _: _transition(
                lambda: fill_cross_selection_state(inputs, state, select=False)
            )
        )

        def distance_command(
            session: Session[CrossSelectionState], params: Mapping[str, object]
        ) -> CrossSelectionState:
            value = decode_parameters(distance_params, params)["min_distance"]
            if not isinstance(value, float):
                raise InvalidInputError("min_distance must be numeric")
            return distance.execute(session, value)

        def tool_command(
            session: Session[CrossSelectionState], params: Mapping[str, object]
        ) -> CrossSelectionState:
            return tool.execute(session, decode_brush_tool(params))

        def stroke_command(
            session: Session[CrossSelectionState], params: Mapping[str, object]
        ) -> CrossSelectionState:
            return stroke.execute(session, decode_brush_stroke(params))

        def perform_command(
            session: Session[CrossSelectionState], params: Mapping[str, object]
        ) -> CrossSelectionState:
            decode_parameters((), params)
            return perform.execute(session, None)

        def clear_command(
            session: Session[CrossSelectionState], params: Mapping[str, object]
        ) -> CrossSelectionState:
            decode_parameters((), params)
            return clear.execute(session, None)

        def can_finish(state: CrossSelectionState) -> None:
            analyze_cross_selection(inputs, state)

        PluginDefinition.__init__(
            self,
            plugin_id="cross_selection",
            seed=seed,
            commands=(
                Command("set_min_distance", distance_params, distance_command),
                Command("set_tool", TOOL_PARAMS, tool_command),
                Command("stroke", STROKE_PARAMS, stroke_command),
                Command("perform_on_all", (), perform_command),
                Command("clear", (), clear_command),
            ),
            can_finish=can_finish,
            build_result=lambda state: analyze_cross_selection(inputs, state),
        )
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "set_min_distance", distance)
        object.__setattr__(self, "set_tool", tool)
        object.__setattr__(self, "stroke", stroke)
        object.__setattr__(self, "perform_on_all", perform)
        object.__setattr__(self, "clear", clear)


def _transition(compute: Callable[[], CrossSelectionState]) -> CrossSelectionState:
    try:
        return compute()
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
