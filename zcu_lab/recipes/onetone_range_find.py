"""Wide absolute-frequency survey with explicit bounds and dip identities."""

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.form_adapter.sweep import LinRange
from zcu_tools.mcp.measure.recipe_api import FieldEdit, RecipeSession, recipe

from zcu_lab.v2.onetone.range_find.core import DEFAULT_MIN_SEPARATION_MHZ

from .helpers import report_number
from .mux_helpers import ToneRangeInputs


class OnetoneRangeFindInputs(FormModel):
    """Explicit absolute MHz bounds and minimum dip spacing, without inferred bounds.

    min_separation defaults to the analysis owner's 1 MHz. It must be finite
    and positive; invalid values raise ValidationError before opening a tab.
    It affects dip selection, not the acquisition grid or hardware channels.
    """

    target: str | None = Field(
        None, description="Resonator Reset alias; null uses global aliases."
    )
    reuse_tab_id: str | None = Field(
        None, description="Idle matching tab to reset; null opens a tab."
    )
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
        description="Dip names in ascending frequency order; null keeps the Form's captured initial list. Names do not select hardware.",
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


class ResonatorDip(BaseModel):
    name: str
    freq: float | None
    depth: float | None


class RangeFindSummary(BaseModel):
    resonators: list[ResonatorDip]
    writeback_diagnostic: str | None
    identity_diagnostic: str | None


class OnetoneRangeFindReport(FormModel):
    """Frequency-ordered dip estimates and identity/writeback limitations."""

    resonators: list[ResonatorDip] = Field(
        description="Each dip's name, absolute MHz frequency and background-relative depth in a.u."
    )
    identity_diagnostic: str | None = Field(
        default=None,
        description="Owner warning that dip/component pairing needs confirmation; null does not prove identity.",
    )
    writeback_diagnostic: str | None = Field(
        default=None,
        description="Owner reason that some frequency proposals were omitted.",
    )
    writeback_applied: bool = Field(
        default=False,
        description="True only after selected proposals were successfully applied.",
    )
    analysis_failure: str | None = Field(
        default=None, description="Top-level analysis failure; null on completion."
    )


@recipe(
    "onetone_range_find",
    description="Survey explicit absolute MHz bounds, report named dips and identity diagnostics, and ask before frequency writeback.",
)
async def onetone_range_find(
    session: RecipeSession, inputs: OnetoneRangeFindInputs
) -> OnetoneRangeFindReport:
    """Run onetone/range_find, Analyze, ask_writeback, accept selections, Save.

    freq and n are required. Null resonators retains the captured Form initial
    list; names label dips in ascending frequency order, not hardware channels.
    min_separation is finite positive MHz, default 1 MHz. It overrides the
    owner's dip spacing through the same tracked Analyze operation, including
    when reusing a tab. It does not change the acquired frequency grid.
    Expected analysis failures return a reason without writeback. The runtime
    owns cancellation and operation failures. Save once after acquisition.
    """
    tab = session.open_tab(
        "onetone/range_find",
        aliases=None if inputs.target is None else {"resonator": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.apply(
        [
            FieldEdit(
                path=("sweep", "freq"),
                value=LinRange(
                    start=inputs.freq.start, stop=inputs.freq.stop, n=inputs.n
                ),
            ),
            FieldEdit(
                path=("resonators",),
                value=None
                if inputs.resonators is None
                else [name for name in inputs.resonators],
            ),
            FieldEdit(path=("reps",), value=inputs.reps),
            FieldEdit(path=("rounds",), value=inputs.rounds),
        ],
        require_valid=True,
    )
    await tab.run()
    try:
        fit = await tab.analyze(
            "primary",
            result=RangeFindSummary,
            updates={"min_separation": inputs.min_separation},
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
            resonators=[
                ResonatorDip(
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
