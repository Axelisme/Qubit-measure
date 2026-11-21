"""Wide absolute-frequency survey with interactive confirmation before writeback."""

import json
from typing import Annotated

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    Overrides,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    ReuseTabId,
    SingleTarget,
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

    target: Annotated[
        SingleTarget,
        Field(
            description="Resonator binding for the survey readout; omit or pass null to keep the current alias."
        ),
    ]
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    freq: ToneRangeInputs = Field(
        description="Set Readout RF frequency › Start and Stop (MHz) to absolute RF endpoints, not mixer offsets; both are required."
    )
    n: int = Field(
        strict=True,
        ge=2,
        description="Set Readout RF frequency › Expts (count); use at least 11 samples for dip selection.",
    )
    resonators: list[str] = Field(
        min_length=1,
        description="Set Analyze › Resonators (JSON) as a nonempty list of names; the list length gives the expected dip count. Use temporary names if needed.",
    )
    min_separation: (
        Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] | None | Keep
    ) = Field(
        KEEP,
        description="Set Analyze › Min separation (MHz), the minimum dip spacing (>0, initially 1); omit to retain it, or pass null to Analyze.",
    )
    reps: Annotated[int, Field(strict=True, gt=0)] | None | Keep = Field(
        KEEP,
        description="Set Reps (count), shots per frequency per round; omit to retain it, or pass null to the Form.",
    )
    rounds: Annotated[int, Field(strict=True, gt=0)] | None | Keep = Field(
        KEEP,
        description="Set Rounds (count), repetitions of the sweep for averaging; omit to retain it, or pass null to the Form.",
    )


class OnetoneRangeFindReport(RecipeReport):
    """Frequency-ordered dip estimates and pairing uncertainty."""

    resonators: list[ResonatorDipSummary] = Field(
        description="Selected dips: name (null if unassigned), frequency (MHz), depth (a.u.), half-prominence width (MHz) and depth/width (a.u./MHz) when available."
    )
    prominence_factor: float | None = Field(
        default=None,
        description="Actual prominence noise multiplier; null when analysis did not complete.",
    )
    identity_diagnostic: str | None = Field(
        default=None,
        description="Owner warning that dip/component pairing needs confirmation; null does not prove identity.",
    )


@recipe(
    "onetone_range_find",
    description="Survey explicit absolute MHz bounds, await agent confirmation with tab_interact done, then ask before frequency writeback.",
)
async def onetone_range_find(
    session: RecipeSession, inputs: OnetoneRangeFindInputs
) -> OnetoneRangeFindReport:
    """Run, await interactive Done, ask_writeback, accept selections, then Save.

    freq, n and nonempty resonators are required. freq supplies absolute MHz
    bounds and n sets freq.expts, the shared acquisition sample count. Names
    set the expected dip count and seed frequency-order pairing in Analyze,
    not cfg. Use temporary names for unidentified dips. The agent must
    inspect or adjust the picker through tab_interact and send done before this
    recipe asks about writeback. Final assignments may use any frequency order.
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
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        fit = await tab.analyze(
            "primary",
            result=RangeFindSummary,
            interactive=True,
            updates={
                "resonators": json.dumps(inputs.resonators),
                "min_separation": inputs.min_separation,
            },
        )
        if fit.result is None:
            return OnetoneRangeFindReport(resonators=[], analysis_failure=fit.reason)
        answer = await tab.ask_writeback()
        applied = False
        if answer.accepted:
            receipt = tab.accept_writeback(answer.items)
            receipt.raise_for_error()
            applied = bool(receipt.items)
        return OnetoneRangeFindReport(
            resonators=list(fit.result.dips),
            prominence_factor=fit.result.prominence_factor,
            identity_diagnostic=fit.result.identity_diagnostic,
            writeback_applied=applied,
        )
    finally:
        tab.save()


DEFINITION = onetone_range_find
