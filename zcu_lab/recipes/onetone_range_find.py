"""Wide absolute-frequency survey with optional dip naming after discovery."""

import json
from typing import Annotated

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
)

from zcu_lab.v2.onetone.range_find.gui import RangeFindSummary, ResonatorDipSummary

from .mux_helpers import ToneRangeInputs


class OnetoneRangeFindInputs(RecipeInputs):
    """Explicit absolute MHz bounds and minimum dip spacing, without inferred bounds.

    Omitted min_separation retains the analysis owner's value (initially 1 MHz).
    A supplied number must be finite and positive or fail before opening a tab.
    It affects dip selection, not the acquisition grid or hardware channels.
    """

    freq: ToneRangeInputs = Field(
        description="Required absolute MHz start/stop, not mixer offsets. No frequency initial guess is inferred."
    )
    n: int = Field(
        strict=True,
        ge=2,
        description="Required frequency sample count; use at least 11 points for dip analysis.",
    )
    resonators: list[str] | None | Keep = Field(
        KEEP,
        description="Optional Analyze names in ascending frequency order; omission retains the analysis names, an empty list discovers unnamed dips, and null is passed to the analysis owner. Names never enter acquisition cfg.",
    )
    min_separation: (
        Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] | None | Keep
    ) = Field(
        KEEP,
        description="Minimum spacing between selected dips in MHz; finite and > 0. Omission retains the analysis value (initially 1 MHz); null is passed to the owner. Does not affect acquisition.",
    )
    reps: Annotated[int, Field(strict=True, gt=0)] | None | Keep = Field(
        KEEP,
        description="Acquisition repetitions; omission retains the Form value; null clears/disables it.",
    )
    rounds: Annotated[int, Field(strict=True, gt=0)] | None | Keep = Field(
        KEEP,
        description="Averaging rounds; omission retains the Form value; null clears/disables it.",
    )


class OnetoneRangeFindReport(RecipeReport):
    """Frequency-ordered dip estimates and identity/writeback limitations."""

    resonators: list[ResonatorDipSummary] = Field(
        description="Each dip's optional name (null during discovery), absolute MHz frequency and background-relative depth in a.u."
    )
    identity_diagnostic: str | None = Field(
        default=None,
        description="Owner warning that dip/component pairing needs confirmation; null does not prove identity.",
    )
    writeback_diagnostic: str | None = Field(
        default=None,
        description="Owner reason that some frequency proposals were omitted.",
    )


@recipe(
    "onetone_range_find",
    description="Survey explicit absolute MHz bounds, discover unnamed dips or assign analysis names, and ask before frequency writeback.",
)
async def onetone_range_find(
    session: RecipeSession, inputs: OnetoneRangeFindInputs
) -> OnetoneRangeFindReport:
    """Run onetone/range_find, Analyze, ask_writeback, accept selections, Save.

    freq and n are required. freq supplies absolute MHz bounds and n sets
    freq.expts, the shared acquisition sample count. Omitted resonators
    retains the analysis names; an empty list discovers unnamed dips. Supplied
    names label dips in ascending frequency order through Analyze, not cfg.
    A supplied min_separation is finite positive MHz and changes only analysis.
    Omission retains the owner's value, initially 1 MHz. Explicit null is passed
    to the analysis owner. Neither field changes the acquired frequency grid.
    Expected analysis failures return a reason without writeback. The runtime
    owns cancellation and operation failures. Save once after acquisition.

    n → freq.expts sets the absolute-frequency sample count.
    Analysis receives resonators and min_separation as explicit updates.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_tab(
        "onetone/range_find",
        target={"resonator": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.set("freq.start", inputs.freq.start)
    tab.set("freq.stop", inputs.freq.stop)
    tab.set("freq.n", inputs.n)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        fit = await tab.analyze(
            "primary",
            result=RangeFindSummary,
            updates={
                "resonators": inputs.resonators
                if inputs.resonators is None or isinstance(inputs.resonators, Keep)
                else json.dumps(inputs.resonators),
                "min_separation": inputs.min_separation,
            },
        )
        if fit.result is None:
            return OnetoneRangeFindReport(resonators=[], analysis_failure=fit.reason)
        applied = False
        if not isinstance(inputs.resonators, Keep) and inputs.resonators:
            answer = await tab.ask_writeback()
            if answer.accepted:
                receipt = tab.accept_writeback(answer.items)
                receipt.raise_for_error()
                applied = bool(receipt.items)
        return OnetoneRangeFindReport(
            resonators=list(fit.result.dips),
            identity_diagnostic=fit.result.identity_diagnostic,
            writeback_diagnostic=fit.result.diagnostic,
            writeback_applied=applied,
        )
    finally:
        tab.save()


DEFINITION = onetone_range_find
