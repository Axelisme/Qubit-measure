"""TwoTone actions and validated commands for one shared interactive Session."""

from __future__ import annotations

from zcu_tools.analysis.fluxdep.twotone import (
    TwoToneInputs,
    TwoTonePickResult,
    TwoTonePickState,
    TwoToneSettings,
    TwoToneStroke,
    TwoToneTool,
)
from zcu_tools.gui.interactive import Action, PluginDefinition


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
        raise NotImplementedError
