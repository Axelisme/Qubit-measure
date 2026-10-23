"""Common-offset mux resonator spectroscopy with reviewed writeback."""

from typing import Literal

from pydantic import BaseModel, Field, StrictInt
from zcu_tools.mcp.measure.recipe_api import (
    FieldEdit,
    FitQuality,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)

from zcu_lab.v2.mux.onetone.gui import MuxOnetoneSummary, MuxOnetoneTone

from .mux_helpers import ToneRangeInputs


class MuxOnetoneInputs(RecipeInputs):
    """Experiment inputs; null retains captured Form values, overrides apply last."""

    participants: list[StrictInt] | None = Field(
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

    model_type: Literal["hm", "t", "auto"] | None = Field(
        None, description="Resonator fit model. Null retains analysis settings."
    )
    fit_bg_amp_slope: bool | None = Field(
        None,
        description="Fit the amplitude background slope. Null retains analysis settings.",
    )
    fit_bg_phase_curvature: bool | None = Field(
        None,
        description="Fit background phase curvature. Null retains analysis settings.",
    )
    edelay: float | None = Field(
        None,
        description="Manual cable-delay seed in inverse MHz. Null retains analysis settings.",
    )


class CaptureRequest(BaseModel):
    """Read the Reset's fixed identity bindings, not live global aliases."""

    paths: list[str] = ["$mux.resonators"]
    module_refs: list[str] = []


class ResonatorCapture(BaseModel):
    """Fixed Reset parameter reply: ordered resonator names or unavailable bindings."""

    values: dict[str, list[str] | None]


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


def tone_report(
    tone: MuxOnetoneTone, names: list[str] | None, diagnostic: str | None
) -> MuxOnetoneToneReport:
    """Pair the native original index only when the owner accepts the bindings."""
    name = None
    if names is not None and diagnostic is None and 0 <= tone.index < len(names):
        name = names[tone.index]
    return MuxOnetoneToneReport(
        index=tone.index,
        label=tone.label,
        status=tone.status,
        fit_quality=tone.fit_quality,
        reason=tone.reason,
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
    """Common-offset mux resonator spectroscopy with reviewed writeback.

    Select library then inline sources; null keeps captured Form values.
    Overrides run last. Active readout tones share a common frequency-offset scan.
    Readout identities remain required for every tone.
    n sets the common df Expts; df uses MHz offsets, reps/rounds set averaging.
    Labels are display-only. Participant indices retain original readout positions.
    Analyze the resonator model/background/delay settings, then ask before writeback.
    Save raw and completed artifacts after Run, including expected fit failures.
    Operation errors propagate without rolling back earlier edits or saves.
    """
    tab = session.open_recipe_tab("mux/onetone", inputs, target_alias="mux")
    capture = tab.call("tab.parameter_read", CaptureRequest(), result=ResonatorCapture)
    names = capture.values.get("$mux.resonators")
    tab.select_sources(inputs)

    tab.apply_inputs(
        inputs,
        {
            ("sweep", "df", "start"): None if inputs.df is None else inputs.df.start,
            ("sweep", "df", "stop"): None if inputs.df is None else inputs.df.stop,
            ("sweep", "df", "n"): inputs.n,
            ("reps",): inputs.reps,
            ("rounds",): inputs.rounds,
        },
        edits=[
            FieldEdit(
                path=("participants",),
                value=None if inputs.participants is None else [*inputs.participants],
            )
        ],
    )
    await tab.run()
    try:
        fit = await tab.analyze(
            "primary",
            updates={
                key: value
                for key, value in {
                    "model_type": inputs.model_type,
                    "fit_bg_amp_slope": inputs.fit_bg_amp_slope,
                    "fit_bg_phase_curvature": inputs.fit_bg_phase_curvature,
                    "edelay": inputs.edelay,
                }.items()
                if value is not None
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
