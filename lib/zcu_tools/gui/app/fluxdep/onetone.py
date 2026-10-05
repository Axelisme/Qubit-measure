"""Fluxdep one-tone commands against a captured-input interactive session."""

from __future__ import annotations

from zcu_tools.analysis.fluxdep.onetone import (
    OneToneInputs,
    OneTonePickResult,
    OneTonePickState,
)
from zcu_tools.gui.interactive.plugin import Action, PluginDefinition


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
        raise NotImplementedError
