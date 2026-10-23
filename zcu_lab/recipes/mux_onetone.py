"""Common-offset mux resonator spectroscopy with reviewed writeback."""

from typing import Literal

from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.mcp.measure.recipe_api import FieldEdit, RecipeSession, recipe

from .helpers import report_number
from .mux_helpers import MuxInline, MuxLibrary, ToneRangeInputs, mux_overrides


class MuxOnetoneInputs(FormModel):
    """Ordered source selections, common MHz offsets and acquisition counts."""

    target: str | None = Field(
        None, description="Mux Reset alias; null uses global aliases."
    )
    reuse_tab_id: str | None = Field(
        None, description="Idle matching tab to reset; null opens a tab."
    )
    participants: list[int] | None = Field(
        None,
        description="Original tone indices in execution order; null keeps the captured Form selection.",
    )
    df: ToneRangeInputs | None = Field(
        None,
        description="Common MHz offset start/stop added to every base tone frequency; null keeps the Form range.",
    )
    n: int | None = Field(
        None,
        strict=True,
        ge=2,
        description="Shared offset sample count; null keeps Expts.",
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
    library: list[MuxLibrary] | None = Field(
        None, description="Ordered captured module selections; null keeps sources."
    )
    inline: list[MuxInline] | None = Field(
        None,
        description="Ordered captured module copies, ancestors first; null keeps sources.",
    )
    overrides: list[FieldEdit] | None = Field(
        None,
        description="Ordered scalar edits applied last; integers are full readout tone indices.",
    )


class CaptureRequest(BaseModel):
    """Read the Reset's fixed identity bindings, not live global aliases."""

    paths: list[str] = ["$mux.resonators"]
    module_refs: list[str] = []


class ResonatorCapture(BaseModel):
    values: dict[str, list[str] | None]


class QualityIssue(BaseModel):
    path: str
    reason: str


class FitQuality(BaseModel):
    """Owner diagnostics without an authored acceptance threshold."""

    r2: float | None
    normalized_residual_rms: float | None
    relative_parameter_errors: dict[str, float | None]
    invalid: list[QualityIssue]


class MuxOnetoneTone(BaseModel):
    """Native original tone identity and fit result or failure."""

    index: int
    label: str
    status: Literal["success", "failed"]
    freq: float | None = None
    kappa: float | None = None
    fit_quality: FitQuality | None = None
    reason: str | None = None


class MuxOnetoneSummary(BaseModel):
    tones: list[MuxOnetoneTone]
    resonators_diagnostic: str | None


class MuxOnetoneToneReport(MuxOnetoneTone):
    name: str | None = Field(
        description="Resonator bound to the original index in the fixed capture, never inferred from label."
    )


class MuxOnetoneReport(FormModel):
    """Native fits, captured identities and the explicit writeback outcome."""

    tones: list[MuxOnetoneToneReport] = Field(
        description="Fits in native analysis order, retaining failed tones."
    )
    resonators_diagnostic: str | None = Field(
        default=None,
        description="Owner reason that resonator identity or scalar writeback is unavailable.",
    )
    writeback_applied: bool = Field(
        default=False,
        description="True only when selected items were successfully applied.",
    )
    analysis_failure: str | None = Field(
        default=None, description="Top-level analysis failure, or null on completion."
    )


def tone_report(
    tone: MuxOnetoneTone, names: list[str] | None, diagnostic: str | None
) -> MuxOnetoneToneReport:
    """Pair the native original index only when the owner accepts the bindings."""
    name = None
    if names is not None and diagnostic is None and 0 <= tone.index < len(names):
        name = names[tone.index]
    return MuxOnetoneToneReport(
        **tone.model_dump(exclude={"freq", "kappa"}),
        freq=report_number(tone.freq),
        kappa=report_number(tone.kappa),
        name=name,
    )


@recipe(
    "mux_onetone",
    description="Scan common MHz offsets for selected mux resonators, report fits and quality, and ask before writeback.",
)
async def mux_onetone(
    session: RecipeSession, inputs: MuxOnetoneInputs
) -> MuxOnetoneReport:
    """Run mux/onetone, Analyze, ask_writeback, accept only selected items, Save.

    Select library then inline sources. Map participants to original indices in
    the fixed Reset capture, apply df/n and counts, then overrides. Unknown or
    ambiguous names return needs_parameters with legal names before acquisition.
    Null keeps Form values. Labels do not establish resonator identity. All
    fields must validate before Run. Expected analysis failures save without
    writeback; cancellation and other operation errors belong to the runtime.
    """
    tab = session.open_tab(
        "mux/onetone",
        aliases=None if inputs.target is None else {"mux": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    capture = tab.call("tab.parameter_read", CaptureRequest(), result=ResonatorCapture)
    names = capture.values.get("$mux.resonators")
    for selection in inputs.library or ():
        tab.select(selection.path, ref=selection.key)
    for copy in inputs.inline or ():
        tab.select(copy.path, inline=True)

    edits = [
        FieldEdit(
            path=("active_tones",),
            value=None if inputs.participants is None else list(inputs.participants),
        )
    ]
    if inputs.df is not None:
        edits.extend(
            [
                FieldEdit(path=("sweep", "df", "start"), value=inputs.df.start),
                FieldEdit(path=("sweep", "df", "stop"), value=inputs.df.stop),
            ]
        )
    edits.extend(
        [
            FieldEdit(path=("sweep", "df", "n"), value=inputs.n),
            FieldEdit(path=("reps",), value=inputs.reps),
            FieldEdit(path=("rounds",), value=inputs.rounds),
        ]
    )
    edits.extend(mux_overrides(inputs.overrides))
    tab.apply(edits, require_valid=True)
    await tab.run()
    try:
        fit = await tab.analyze("primary", result=MuxOnetoneSummary)
        if fit.result is None:
            return MuxOnetoneReport(tones=[], analysis_failure=fit.reason)
        answer = await tab.ask_writeback()
        applied = False
        if answer.accepted:
            receipt = tab.accept_writeback(answer.items)
            receipt.raise_for_error()
            applied = bool(receipt.items)
        tones = [
            tone_report(tone, names, fit.result.resonators_diagnostic)
            for tone in fit.result.tones
        ]
        return MuxOnetoneReport(
            tones=tones,
            resonators_diagnostic=fit.result.resonators_diagnostic,
            writeback_applied=applied,
        )
    finally:
        tab.save()


DEFINITION = mux_onetone
