"""Acquire a single-carrier Raman scan using operator spectroscopy and readout."""

from typing import Literal

from pydantic import Field
from zcu_tools.analysis.singleshot import Centroid3Summary
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
    report_number,
)


class RamanGFInputs(RecipeInputs):
    """Raman controls; omitted fields retain the Form, null goes to its owner."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    readout_ref: str | None | Keep = Field(
        KEEP,
        description="Set Readout to a library module with saved centroid3 calibration.",
    )
    drive_ref: str | None | Keep = Field(
        KEEP,
        description="Set Raman drive to the selected library pulse.",
    )
    waveform_asset: str | None | Keep = Field(
        KEEP,
        description="Set Raman drive › Waveform › Arbitrary asset to the sampled envelope key.",
    )
    carrier_mhz: float | None | Keep = Field(
        KEEP,
        description="Set Carrier center (MHz) to half the measured f03 minus f32 difference.",
    )
    scan_mode: Literal["frequency", "gain", "frequency_gain"] | None | Keep = Field(
        KEEP,
        description="Set Scan mode to frequency, gain or frequency_gain.",
    )
    detuning_start_mhz: float | None | Keep = Field(
        KEEP,
        description="Set Two-photon detuning › Start (MHz); positive raises the carrier by half this value.",
    )
    detuning_stop_mhz: float | None | Keep = Field(
        KEEP,
        description="Set Two-photon detuning › Stop (MHz); positive raises the carrier by half this value.",
    )
    detuning_n: int | None | Keep = Field(
        KEEP,
        description="Set Two-photon detuning › Expts (count).",
    )
    gain_start: float | None | Keep = Field(
        KEEP,
        description="Set Drive gain › Start (a.u.); frequency mode holds this signed gain.",
    )
    gain_stop: float | None | Keep = Field(
        KEEP,
        description="Set Drive gain › Stop (a.u.), a signed DAC full-scale fraction.",
    )
    gain_n: int | None | Keep = Field(
        KEEP,
        description="Set Drive gain › Expts (count).",
    )
    shots: int | None | Keep = Field(
        KEEP,
        description="Set Shots per point (count), classified into g/e/f labels.",
    )
    relax_delay_us: float | None | Keep = Field(
        KEEP,
        description="Set Relax delay (µs), the wait after each shot.",
    )


class RamanGFReport(RecipeReport):
    """Applied readout matrix quality; population arrays stay in analysis artifacts."""

    condition_number: float | None = Field(
        default=None,
        description="2-norm condition number of the applied readout matrix; null without correction or if unrepresentable.",
    )
    minimum_singular_value: float | None = Field(
        default=None,
        description="Smallest singular value of the applied readout matrix; null without correction.",
    )


@recipe(
    "raman_gf",
    description="Run a single-tone two-photon g/f scan, analyze g/e/f counts and save counts and analysis.",
)
async def raman_gf(session: RecipeSession, inputs: RamanGFInputs) -> RamanGFReport:
    """Play one Raman pulse then readout at each scan point, analyze and save.

    The operator supplies carrier_mhz from spectroscopy in MHz, never from a
    simulator model. Detuning is 2*f_drive - 2*carrier_mhz; positive detuning
    raises the single carrier by detuning/2. Frequency mode holds gain_start;
    gain mode holds detuning_start_mhz; frequency_gain scans both axes.
    detuning_n/gain_n set Form n, lowered to cfg expts. Gain is a DAC full-scale
    fraction (a.u.), not a calibrated rotation angle. Relaxation timing uses µs.

    Omission (Python KEEP, not a JSON value) retains the Form value and source.
    Explicit null is passed unchanged to the Form owner, including overrides.
    An explicit waveform asset converts drive and waveform to inline without
    replacing their family; choose an arbitrary waveform source first.
    Ordered overrides follow the named inputs. Missing/invalid inputs are
    reported by the Form before Run. Analyze uses classified counts from this
    run and applies its saved centroid3 matrix only when usable. It never reads
    live calibration or simulator truth.
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
    tab.apply_overrides(inputs.overrides)
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
