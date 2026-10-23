"""Wide absolute-frequency survey with optional dip naming after discovery."""

import json

from pydantic import Field
from zcu_tools.gui.form_adapter.sweep import LinRange
from zcu_tools.mcp.measure.recipe_api import (
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)

from zcu_lab.v2.onetone.range_find.core import DEFAULT_MIN_SEPARATION_MHZ
from zcu_lab.v2.onetone.range_find.gui import RangeFindSummary, ResonatorDipSummary

from .mux_helpers import ToneRangeInputs


class OnetoneRangeFindInputs(RecipeInputs):
    """Explicit absolute MHz bounds and minimum dip spacing, without inferred bounds.

    min_separation defaults to the analysis owner's 1 MHz. It must be finite
    and positive; invalid values raise ValidationError before opening a tab.
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
    resonators: list[str] | None = Field(
        None,
        description="Optional Analyze names in ascending frequency order; omitted, null or empty discovers unnamed dips without writeback. Names never enter acquisition cfg.",
    )
    min_separation: float = Field(
        default=DEFAULT_MIN_SEPARATION_MHZ,
        strict=True,
        gt=0,
        allow_inf_nan=False,
        description="Minimum spacing between selected dips in MHz; finite and > 0, default 1 MHz. Applied to this analysis, not acquisition.",
    )
    reps: int | None = Field(
        None,
        strict=True,
        gt=0,
        description="Acquisition repetitions; null keeps the Form value.",
    )
    rounds: int | None = Field(
        None,
        strict=True,
        gt=0,
        description="Averaging rounds; null keeps the Form value.",
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
    sweep.freq.expts, the shared acquisition sample count. Omitted, null or empty
    resonators discovers unnamed dips without writeback. Otherwise names label
    dips in ascending frequency order through Analyze, not acquisition cfg.
    min_separation is finite positive MHz, default 1 MHz. It overrides the
    owner's dip spacing through the same tracked Analyze operation, including
    when reusing a tab. It does not change the acquired frequency grid.
    Expected analysis failures return a reason without writeback. The runtime
    owns cancellation and operation failures. Save once after acquisition.

    n → sweep.freq.expts sets the absolute-frequency sample count.
    Analysis receives resonators and min_separation as explicit updates.
    Missing Form values stop before acquisition; completed Run data is saved even if analysis fails.
    """
    tab = session.open_recipe_tab(
        "onetone/range_find", inputs, target_alias="resonator"
    )
    tab.select_sources(inputs)
    tab.apply_inputs(
        inputs,
        {
            ("sweep", "freq"): LinRange(
                start=inputs.freq.start, stop=inputs.freq.stop, n=inputs.n
            ),
            ("reps",): inputs.reps,
            ("rounds",): inputs.rounds,
        },
    )
    await tab.run()
    try:
        fit = await tab.analyze(
            "primary",
            result=RangeFindSummary,
            updates={
                "resonators": json.dumps(inputs.resonators or []),
                "min_separation": inputs.min_separation,
            },
        )
        if fit.result is None:
            return OnetoneRangeFindReport(resonators=[], analysis_failure=fit.reason)
        applied = False
        if inputs.resonators:
            answer = await tab.ask_writeback()
            if answer.accepted:
                receipt = tab.accept_writeback(answer.items)
                receipt.raise_for_error()
                applied = bool(receipt.items)
        return OnetoneRangeFindReport(
            resonators=[
                ResonatorDipSummary(
                    name=dip.name,
                    freq=report_number(dip.freq),
                    depth=report_number(dip.depth),
                )
                for dip in fit.result.resonators
            ],
            identity_diagnostic=fit.result.identity_diagnostic,
            writeback_diagnostic=fit.result.writeback_diagnostic,
            writeback_applied=applied,
        )
    finally:
        tab.save()


DEFINITION = onetone_range_find
