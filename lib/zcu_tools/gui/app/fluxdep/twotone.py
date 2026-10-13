"""TwoTone actions and validated commands for one shared interactive Session.

SET_SETTINGS_PARAMS declares optional dimensionless detector settings, including
the method-dependent sigma limit. Commands require at least one non-null value.
Remote consumers reuse these declarations without constructing a plugin.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from zcu_tools.analysis.fluxdep.stroke import BrushStroke, BrushTool
from zcu_tools.analysis.fluxdep.twotone import (
    GAUSSIAN_SIGMA_MIN,
    SIGMA_MAX,
    SIGMA_MIN,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    TwoToneInputs,
    TwoTonePickResult,
    TwoTonePickState,
    TwoToneSettings,
    analyze_twotone_pick,
    fill_twotone_state,
    make_twotone_state,
    set_twotone_settings,
    set_twotone_tool,
    stroke_twotone_state,
)
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

SET_SETTINGS_PARAMS = (
    ParamSpec(
        "threshold",
        JsonType.NUMBER,
        required=False,
        minimum=THRESHOLD_MIN,
        maximum=THRESHOLD_MAX,
        description=f"Dimensionless detector threshold [{THRESHOLD_MIN:g},{THRESHOLD_MAX:g}].",
    ),
    ParamSpec(
        "sigma",
        JsonType.NUMBER,
        required=False,
        minimum=SIGMA_MIN,
        maximum=SIGMA_MAX,
        description=(
            f"Dimensionless wavelet sigma [{SIGMA_MIN:g},{SIGMA_MAX:g}]; "
            f"Gaussian sigma [{GAUSSIAN_SIGMA_MIN:g},{SIGMA_MAX:g}]."
        ),
    ),
    ParamSpec(
        "smooth_method",
        JsonType.STRING,
        required=False,
        enum=("wavelet", "gaussian"),
        description="Smoothing method; Gaussian requires the sigma range declared above.",
    ),
)


class TwoTonePickPlugin(PluginDefinition[TwoTonePickState, TwoTonePickResult]):
    """Share brush/detector transitions between GUI and validated commands.

    inputs is the captured numerical input. set_settings, stroke,
    perform_on_all and clear establish undo; set_tool preserves its history.
    Actions translate numerical ValueError to InvalidInputError without commit.
    Commands also translate RemoteError; input gates are inherited from Session.
    """

    inputs: TwoToneInputs
    set_settings: Action[TwoTonePickState, TwoToneSettings]
    set_tool: Action[TwoTonePickState, BrushTool]
    stroke: Action[TwoTonePickState, BrushStroke]
    perform_on_all: Action[TwoTonePickState, None]
    clear: Action[TwoTonePickState, None]

    def __init__(
        self, inputs: TwoToneInputs, threshold: float = 1.0, brush_width: float = 0.05
    ) -> None:
        """Declare twotone_pick commands on an empty selection seed.

        set_settings accepts optional threshold/sigma/smooth_method; set_tool
        accepts optional width/mode. Both require at least one non-null value.
        stroke requires vertices NUMBER_PAIRS, width NUMBER, mode select/erase.
        perform_on_all uses current mode; clear always erases. Unknown fields,
        missing/invalid values and invalid domains raise InvalidInputError.
        Invalid seeds raise ValueError. Finish computes sorted native points
        and closes input, including for a successful empty selection.
        """
        seed = make_twotone_state(inputs, threshold=threshold, width=brush_width)
        # GUI sampling starts fresh; the numerical/Notebook seed remains all-selected.
        seed.mask.fill(False)

        settings: Action[TwoTonePickState, TwoToneSettings] = Action(
            lambda state, params: _transition(
                lambda: set_twotone_settings(inputs, state, params)
            )
        )
        tool: Action[TwoTonePickState, BrushTool] = Action(
            lambda state, params: _transition(
                lambda: set_twotone_tool(inputs, state, params)
            ),
            record_undo=False,
        )
        stroke: Action[TwoTonePickState, BrushStroke] = Action(
            lambda state, params: _transition(
                lambda: stroke_twotone_state(inputs, state, params)
            )
        )
        perform: Action[TwoTonePickState, None] = Action(
            lambda state, _: _transition(
                lambda: fill_twotone_state(inputs, state, select=state.mode == "select")
            )
        )
        clear: Action[TwoTonePickState, None] = Action(
            lambda state, _: _transition(
                lambda: fill_twotone_state(inputs, state, select=False)
            )
        )

        def set_settings_command(
            session: Session[TwoTonePickState], params: Mapping[str, object]
        ) -> TwoTonePickState:
            return settings.execute(
                session,
                _settings_payload(decode_parameters(SET_SETTINGS_PARAMS, params)),
            )

        def set_tool_command(
            session: Session[TwoTonePickState], params: Mapping[str, object]
        ) -> TwoTonePickState:
            return tool.execute(session, decode_brush_tool(params))

        def stroke_command(
            session: Session[TwoTonePickState], params: Mapping[str, object]
        ) -> TwoTonePickState:
            return stroke.execute(session, decode_brush_stroke(params))

        def perform_command(
            session: Session[TwoTonePickState], params: Mapping[str, object]
        ) -> TwoTonePickState:
            decode_parameters((), params)
            return perform.execute(session, None)

        def clear_command(
            session: Session[TwoTonePickState], params: Mapping[str, object]
        ) -> TwoTonePickState:
            decode_parameters((), params)
            return clear.execute(session, None)

        def can_finish(state: TwoTonePickState) -> None:
            analyze_twotone_pick(inputs, state)

        PluginDefinition.__init__(
            self,
            plugin_id="twotone_pick",
            seed=seed,
            commands=(
                Command("set_settings", SET_SETTINGS_PARAMS, set_settings_command),
                Command("set_tool", TOOL_PARAMS, set_tool_command),
                Command("stroke", STROKE_PARAMS, stroke_command),
                Command("perform_on_all", (), perform_command),
                Command("clear", (), clear_command),
            ),
            can_finish=can_finish,
            build_result=lambda state: analyze_twotone_pick(inputs, state),
        )
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "set_settings", settings)
        object.__setattr__(self, "set_tool", tool)
        object.__setattr__(self, "stroke", stroke)
        object.__setattr__(self, "perform_on_all", perform)
        object.__setattr__(self, "clear", clear)


def _transition(compute: Callable[[], TwoTonePickState]) -> TwoTonePickState:
    try:
        return compute()
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc


def _settings_payload(values: Mapping[str, object]) -> TwoToneSettings:
    threshold, sigma, method = (
        values["threshold"],
        values["sigma"],
        values["smooth_method"],
    )
    if threshold is not None and not isinstance(threshold, float):
        raise InvalidInputError("threshold must be numeric")
    if sigma is not None and not isinstance(sigma, float):
        raise InvalidInputError("sigma must be numeric")
    if method is not None and method not in ("wavelet", "gaussian"):
        raise InvalidInputError("smooth_method must be wavelet or gaussian")
    return TwoToneSettings(
        threshold,
        sigma,
        "wavelet"
        if method == "wavelet"
        else "gaussian"
        if method == "gaussian"
        else None,
    )
