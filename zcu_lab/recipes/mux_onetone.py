"""Common-offset mux resonator spectroscopy with reviewed writeback."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StrictInt, TypeAdapter
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    FitQuality,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)

from zcu_lab.v2.mux.onetone.gui import (
    MuxOnetoneFailureSummary,
    MuxOnetoneSummary,
)

from .mux_helpers import ToneRangeInputs


class MuxOnetoneInputs(RecipeInputs):
    """Omitted Form inputs retain values; null clears them; overrides apply last."""

    participants: list[StrictInt] | None | Keep = Field(
        default=KEEP,
        description="Original tone indices in execution order; omission keeps the Form selection; null clears it.",
    )
    df: ToneRangeInputs | None | Keep = Field(
        default=KEEP,
        description="Common MHz offset start/stop added to every base tone frequency; omission keeps the Form range; null clears its bounds.",
    )
    n: Annotated[int, Field(strict=True, ge=2)] | None | Keep = Field(
        default=KEEP,
        description="Shared offset sample count; omission keeps Expts; null clears it.",
    )
    reps: Annotated[int, Field(strict=True, gt=0)] | None | Keep = Field(
        default=KEEP,
        description="Acquisition repetitions; omission keeps the Form value; null clears it.",
    )
    rounds: Annotated[int, Field(strict=True, gt=0)] | None | Keep = Field(
        default=KEEP,
        description="Averaging rounds; omission keeps the Form value; null clears it.",
    )

    model_type: Literal["hm", "t", "auto"] | None | Keep = Field(
        default=KEEP,
        description="Resonator fit model. Omission keeps analysis settings; null is passed to the analysis owner.",
    )
    fit_bg_amp_slope: bool | None | Keep = Field(
        default=KEEP,
        description="Fit the amplitude background slope. Omission keeps analysis settings; null is passed to the analysis owner.",
    )
    fit_bg_phase_curvature: bool | None | Keep = Field(
        default=KEEP,
        description="Fit background phase curvature. Omission keeps analysis settings; null is passed to the analysis owner.",
    )
    edelay: float | None | Keep = Field(
        default=KEEP,
        description="Manual cable-delay seed in inverse MHz. Omission keeps analysis settings; null is passed to the analysis owner.",
    )


class MuxOnetoneToneReport(BaseModel):
    """Original index/label, MHz estimates, core quality or the native fit reason."""

    index: int
    label: str
    status: Literal["success", "failed"]
    freq: float | None = None
    kappa: float | None = None
    fit_quality: FitQuality | None = None
    reason: str | None = None

    name: str | None = Field(
        description="Resonator bound to the original index in the fixed capture, never inferred from label."
    )


class MuxOnetoneReport(RecipeReport):
    """Native ordered estimates/failures and the confirmed writeback outcome."""

    tones: list[MuxOnetoneToneReport] = Field(
        description="Fits in native analysis order, retaining failed tones."
    )
    resonators_diagnostic: str | None = Field(
        default=None,
        description="Owner reason that resonator identity or scalar writeback is unavailable.",
    )


@recipe(
    "mux_onetone",
    description="Scan common MHz offsets for selected mux resonators, report fits and quality, and ask before writeback.",
)
async def mux_onetone(
    session: RecipeSession, inputs: MuxOnetoneInputs
) -> MuxOnetoneReport:
    """Common-offset mux resonator spectroscopy with reviewed writeback.

    Open captures Form sources; omitted inputs keep values and null clears them.
    Overrides run last. Active readout tones share a common frequency-offset scan.
    Readout identities remain required for every tone.
    n sets the common df Expts; df uses MHz offsets, reps/rounds set averaging.
    Labels are display-only. Participant indices retain original readout positions.
    Analyze the resonator model/background/delay settings, then ask before writeback.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_tab(
        "mux/onetone", target={"mux": inputs.target}, reuse=inputs.reuse_tab_id
    )
    tab.set("participants", inputs.participants)
    if isinstance(inputs.df, ToneRangeInputs):
        tab.set("df.start", inputs.df.start)
        tab.set("df.stop", inputs.df.stop)
    elif inputs.df is None:
        tab.set("df.start", None)
        tab.set("df.stop", None)
    tab.set("df.n", inputs.n)
    tab.set("reps", inputs.reps)
    tab.set("rounds", inputs.rounds)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    names = TypeAdapter(list[str] | None).validate_python(
        tab.read_parameter("$mux.resonators", missing_ok=True)
    )

    await tab.run()
    try:
        fit = await tab.analyze(
            "primary",
            updates={
                "model_type": inputs.model_type,
                "fit_bg_amp_slope": inputs.fit_bg_amp_slope,
                "fit_bg_phase_curvature": inputs.fit_bg_phase_curvature,
                "edelay": inputs.edelay,
            },
            result=MuxOnetoneSummary,
        )
        if fit.result is None:
            return MuxOnetoneReport(tones=[], analysis_failure=fit.reason)

        answer = await tab.ask_writeback()
        applied = False
        if answer.accepted:
            receipt = tab.accept_writeback(answer.items)
            receipt.raise_for_error()
            applied = bool(receipt.items)

        tones: list[MuxOnetoneToneReport] = []
        for tone in fit.result.outcomes:
            name = None
            if (
                names is not None
                and fit.result.resonators_diagnostic is None
                and 0 <= tone.index < len(names)
            ):
                name = names[tone.index]
            if isinstance(tone, MuxOnetoneFailureSummary):
                tones.append(
                    MuxOnetoneToneReport(
                        index=tone.index,
                        label=tone.label,
                        status="failed",
                        reason=tone.reason,
                        name=name,
                    )
                )
                continue
            tones.append(
                MuxOnetoneToneReport(
                    index=tone.index,
                    label=tone.label,
                    status="success",
                    fit_quality=tone.fit.fit_quality,
                    freq=report_number(tone.fit.freq),
                    kappa=report_number(tone.fit.fwhm),
                    name=name,
                )
            )
        return MuxOnetoneReport(
            tones=tones,
            resonators_diagnostic=fit.result.resonators_diagnostic,
            writeback_applied=applied,
        )
    finally:
        tab.save()


DEFINITION = mux_onetone
