"""Acquire a single-carrier Raman scan using operator spectroscopy and readout."""

from typing import Literal

from pydantic import Field
from zcu_tools.analysis.singleshot import Centroid3Summary
from zcu_tools.mcp.measure.recipe_api import (
    KEEP,
    Keep,
    RecipeInputs,
    RecipeReport,
    RecipeSession,
    recipe,
    report_number,
)


class RamanGFInputs(RecipeInputs):
    """Raman controls; omitted fields retain the Form, null goes to its owner."""

    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Readout library key with acquired centroid3 calibration.",
    )
    drive_ref: str | None | Keep = Field(
        KEEP,
        description="Single Raman drive library key.",
    )
    waveform_asset: str | None | Keep = Field(
        KEEP,
        description="Sampled arbitrary waveform asset key, such as an operator-prepared sqrt(cos) envelope.",
    )
    carrier_mhz: float | None | Keep = Field(
        KEEP,
        description="Operator-supplied (measured f03 - measured f32)/2 in MHz. No model lookup.",
    )
    scan_mode: Literal["frequency", "area", "frequency_area"] | None | Keep = Field(
        KEEP,
        description="Frequency, area or Cartesian frequency-area scan.",
    )
    detuning_start_mhz: float | None | Keep = Field(
        KEEP,
        description="Two-photon detuning start in MHz: 2*f_drive - 2*carrier_mhz.",
    )
    detuning_stop_mhz: float | None | Keep = Field(
        KEEP,
        description="Two-photon detuning stop in MHz; positive raises the single carrier by half the detuning.",
    )
    detuning_n: int | None | Keep = Field(
        KEEP,
        description="Detuning sample count, Form detuning.n to cfg detuning.expts.",
    )
    gain_start: float | None | Keep = Field(
        KEEP,
        description="Nonnegative drive-gain start in native units; frequency mode holds this gain.",
    )
    gain_stop: float | None | Keep = Field(
        KEEP,
        description="Native nonnegative gain stop; area = gain squared times envelope intensity integral (us), not angle.",
    )
    gain_n: int | None | Keep = Field(
        KEEP,
        description="Gain sample count, Form gain.n to cfg gain.expts.",
    )
    shots: int | None | Keep = Field(
        KEEP,
        description="Repetitions per point, classified into saved g/e/f counts.",
    )
    relax_delay_us: float | None | Keep = Field(
        KEEP,
        description="Relaxation interval in us for ground preparation.",
    )


class RamanGFReport(RecipeReport):
    """Saved-readout matrix quality; population arrays remain in analysis artifacts."""

    condition_number: float | None = Field(
        default=None,
        description="Saved training confusion matrix 2-norm condition number; null if analysis failed.",
    )
    minimum_singular_value: float | None = Field(
        default=None,
        description="Saved training confusion matrix minimum singular value; null if analysis failed.",
    )


@recipe(
    "raman_gf",
    description="Run a single-tone two-photon g/f scan, analyze g/e/f counts and save counts and analysis.",
)
async def raman_gf(session: RecipeSession, inputs: RamanGFInputs) -> RamanGFReport:
    """Play one sampled Raman pulse then readout at each scan point, analyze and save.

    The operator supplies carrier_mhz from spectroscopy in MHz, never from a
    simulator model. Detuning is 2*f_drive - 2*carrier_mhz; positive detuning
    raises the single carrier by detuning/2. Frequency mode holds gain_start;
    area mode holds detuning_start_mhz; frequency_area scans both axes.
    detuning_n/gain_n set Form n, lowered to cfg expts. Gain is in native units;
    source-envelope area is gain^2 us. Relaxation timing uses us.

    Omission (Python KEEP, not a JSON value) retains the Form value and source.
    Explicit null is passed unchanged to the Form owner, including overrides.
    An explicit waveform asset converts drive and waveform to inline without
    replacing their family; choose an arbitrary waveform source first.
    Ordered overrides follow the named inputs. Missing/invalid inputs are
    reported by the Form before Run. Analyze uses only this run's saved
    readout centroid3 calibration, not live calibration or simulator truth.
    No writeback is proposed or requested. Analysis failure still saves g/e/f counts;
    owner errors propagate after Save. Save receipts identify failed outputs.
    """
    tab = session.open_tab(
        "twotone/raman_gf",
        target={"qubit": inputs.target},
        reuse=inputs.reuse_tab_id,
    )
    tab.select("readout", inputs.readout_ref)
    tab.select("drive", inputs.drive_ref)
    if not isinstance(inputs.waveform_asset, Keep):
        tab.command("drive", "to_inline")
        tab.command("drive.waveform", "to_inline")
    tab.set("drive.waveform.data", inputs.waveform_asset)
    tab.set("carrier_mhz", inputs.carrier_mhz)
    tab.set("scan_mode", inputs.scan_mode)
    tab.set("detuning.start", inputs.detuning_start_mhz)
    tab.set("detuning.stop", inputs.detuning_stop_mhz)
    tab.set("detuning.n", inputs.detuning_n)
    tab.set("gain.start", inputs.gain_start)
    tab.set("gain.stop", inputs.gain_stop)
    tab.set("gain.n", inputs.gain_n)
    tab.set("shots", inputs.shots)
    tab.set("relax_delay", inputs.relax_delay_us)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze("primary", result=Centroid3Summary)
        if analysis.result is None:
            return RamanGFReport(analysis_failure=analysis.reason)
        return RamanGFReport(
            condition_number=report_number(analysis.result.condition_number),
            minimum_singular_value=report_number(
                analysis.result.minimum_singular_value
            ),
        )
    finally:
        tab.save()


DEFINITION = raman_gf
