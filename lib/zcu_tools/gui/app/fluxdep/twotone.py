"""TwoTone actions and validated commands for one shared interactive Session."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from zcu_tools.analysis.fluxdep.stroke import BrushPoint
from zcu_tools.analysis.fluxdep.twotone import (
    TwoToneInputs,
    TwoTonePickResult,
    TwoTonePickState,
    TwoToneSettings,
    TwoToneStroke,
    TwoToneTool,
    analyze_twotone_pick,
    fill_twotone_state,
    make_twotone_state,
    set_twotone_settings,
    set_twotone_tool,
    stroke_twotone_state,
)
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.interactive import Action, Command, PluginDefinition, Session
from zcu_tools.gui.remote.errors import RemoteError
from zcu_tools.gui.remote.param_spec import (
    JsonType,
    NumberPairs,
    ParamSpec,
    validate_params,
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
    set_tool: Action[TwoTonePickState, TwoToneTool]
    stroke: Action[TwoTonePickState, TwoToneStroke]
    perform_on_all: Action[TwoTonePickState, None]
    clear: Action[TwoTonePickState, None]

    def __init__(
        self, inputs: TwoToneInputs, threshold: float = 1.0, brush_width: float = 0.05
    ) -> None:
        """Declare twotone_pick commands on an all-selected seed.

        set_settings accepts optional threshold/sigma/smooth_method; set_tool
        accepts optional width/mode. Both require at least one non-null value.
        stroke requires vertices NUMBER_PAIRS, width NUMBER, mode select/erase.
        perform_on_all uses current mode; clear always erases. Unknown fields,
        missing/invalid values and invalid domains raise InvalidInputError.
        Invalid seeds raise ValueError. Finish computes sorted native points
        and closes input, including for a successful empty selection.
        """
        seed = make_twotone_state(inputs, threshold=threshold, width=brush_width)
        settings_params = (
            ParamSpec("threshold", JsonType.NUMBER, required=False),
            ParamSpec("sigma", JsonType.NUMBER, required=False),
            ParamSpec(
                "smooth_method",
                JsonType.STRING,
                required=False,
                enum=("wavelet", "gaussian"),
            ),
        )
        tool_params = (
            ParamSpec("width", JsonType.NUMBER, required=False),
            ParamSpec(
                "mode", JsonType.STRING, required=False, enum=("select", "erase")
            ),
        )
        stroke_params = (
            ParamSpec("vertices", JsonType.NUMBER_PAIRS),
            ParamSpec("width", JsonType.NUMBER),
            ParamSpec("mode", JsonType.STRING, enum=("select", "erase")),
        )
        settings: Action[TwoTonePickState, TwoToneSettings] = Action(
            lambda state, params: _transition(
                lambda: set_twotone_settings(inputs, state, params)
            )
        )
        tool: Action[TwoTonePickState, TwoToneTool] = Action(
            lambda state, params: _transition(
                lambda: set_twotone_tool(inputs, state, params)
            ),
            record_undo=False,
        )
        stroke: Action[TwoTonePickState, TwoToneStroke] = Action(
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
                session, _settings_payload(_decode(settings_params, params))
            )

        def set_tool_command(
            session: Session[TwoTonePickState], params: Mapping[str, object]
        ) -> TwoTonePickState:
            return tool.execute(session, _tool_payload(_decode(tool_params, params)))

        def stroke_command(
            session: Session[TwoTonePickState], params: Mapping[str, object]
        ) -> TwoTonePickState:
            return stroke.execute(
                session, _stroke_payload(_decode(stroke_params, params))
            )

        def perform_command(
            session: Session[TwoTonePickState], params: Mapping[str, object]
        ) -> TwoTonePickState:
            _decode((), params)
            return perform.execute(session, None)

        def clear_command(
            session: Session[TwoTonePickState], params: Mapping[str, object]
        ) -> TwoTonePickState:
            _decode((), params)
            return clear.execute(session, None)

        def can_finish(state: TwoTonePickState) -> None:
            analyze_twotone_pick(inputs, state)

        PluginDefinition.__init__(
            self,
            plugin_id="twotone_pick",
            seed=seed,
            commands=(
                Command("set_settings", settings_params, set_settings_command),
                Command("set_tool", tool_params, set_tool_command),
                Command("stroke", stroke_params, stroke_command),
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


def _decode(
    specs: tuple[ParamSpec, ...], params: Mapping[str, object]
) -> dict[str, object]:
    unknown = params.keys() - {spec.name for spec in specs}
    if unknown:
        raise InvalidInputError(f"unknown parameters: {sorted(unknown)!r}")
    try:
        return validate_params(specs, params)
    except RemoteError as exc:
        raise InvalidInputError(str(exc)) from exc


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


def _tool_payload(values: Mapping[str, object]) -> TwoToneTool:
    width, mode = values["width"], values["mode"]
    if width is not None and not isinstance(width, float):
        raise InvalidInputError("width must be numeric")
    if mode is not None and mode not in ("select", "erase"):
        raise InvalidInputError("mode must be select or erase")
    return TwoToneTool(
        width, "select" if mode == "select" else "erase" if mode == "erase" else None
    )


def _stroke_payload(values: Mapping[str, object]) -> TwoToneStroke:
    vertices, width, mode = values["vertices"], values["width"], values["mode"]
    if not isinstance(vertices, NumberPairs) or not isinstance(width, float):
        raise InvalidInputError("stroke requires vertices and width")
    if not isinstance(mode, str) or (mode != "select" and mode != "erase"):
        raise InvalidInputError("mode must be select or erase")
    return TwoToneStroke(
        tuple(BrushPoint(x, y) for x, y in vertices.values), width, mode
    )
