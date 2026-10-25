"""Composite point calibration: frequency, Rabi including AmpRabi, then readout."""

from dataclasses import dataclass, replace

from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Live
from zcu_tools.plotting.plots import Plots

from zcu_lab.workflows.common.calibration import (
    CalCarry,
    CalibrationInputs,
    CalibrationTunables,
    FrequencyResult,
    RabiResult,
    ReadoutResult,
    calibrate_frequency,
    calibrate_rabi,
    calibrate_readout,
    readout_due,
)


@dataclass
class PointCalibrationResult:
    """Results reached in this point's calibration, without point-record mutation.

    frequency, rabi and readout retain their individual status, reason, scalars
    and native references. None means cancellation prevented that stage from
    being reached. Expected failed/rejected stages return their normal result.
    Unexpected exceptions propagate without returning this aggregate; native
    files already saved by completed stages remain on disk.
    """

    frequency: FrequencyResult | None = None
    rabi: RabiResult | None = None
    readout: ReadoutResult | None = None


async def calibrate_point(
    scope: Scope,
    plots: Plots,
    inputs: CalibrationInputs,
    settings: Live[CalibrationTunables],
    *,
    carry: CalCarry,
    t1_estimate_us: float,
) -> PointCalibrationResult:
    """Calibrate this point's frequency, pi/pi2 and due readout in order.

    inputs supplies this point's pulse templates, flux/index, save identity and
    initially captured frequency policy. settings supplies fresh Rabi/readout
    policies before those stages. carry retains accepted calibration between
    points; t1_estimate_us is the positive history-based relax estimate in us.
    Rabi delegates len_only, amp_only or len_then_amp to calibrate_rabi, including
    AmpRabi where selected. Single-experiment helpers own their prerequisites.
    Cancellation returns results already reached, with later fields None.
    Expected failures/rejections return normal results. Unexpected exceptions
    propagate without returning an aggregate: frequency success then a Rabi bug
    leaves frequency's saved h5 on disk, but not in the caller's point record.
    The result is created and accumulated locally, never passed in by a caller.
    This function never receives or changes point_record.
    """
    result = PointCalibrationResult()
    if scope.cancelled:
        return result
    result.frequency = await calibrate_frequency(
        scope, plots=plots, inputs=inputs, flux=inputs.flux, carry=carry
    )
    if scope.cancelled:
        return result
    result.rabi = await calibrate_rabi(
        scope,
        plots=plots,
        inputs=replace(inputs, tun=settings.current),
        frequency=result.frequency,
        carry=carry,
        t1_estimate_us=t1_estimate_us,
    )
    if scope.cancelled:
        return result
    readout_settings = settings.current
    if not readout_due(inputs.point_index, readout_settings):
        result.readout = ReadoutResult(
            status="skipped",
            reason="Readout flux-point cadence",
            frequency_mhz=None,
            gain=None,
            file=None,
        )
        return result
    result.readout = await calibrate_readout(
        scope,
        plots=plots,
        inputs=replace(inputs, tun=readout_settings),
        rabi=result.rabi,
        carry=carry,
        t1_estimate_us=t1_estimate_us,
    )
    return result
