"""Composite point calibration: frequency, Rabi including AmpRabi, then readout."""

from dataclasses import dataclass, replace

from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.workflows import Live
from zcu_tools.program.v2.modules import PulseReadoutCfg

from zcu_lab.workflows.common.calibration import (
    CalCarry,
    Calibrated,
    CalibrationSource,
    FrequencyResult,
    RabiResult,
    ReadoutResult,
    calibrate_frequency,
    calibrate_rabi,
    calibrate_readout,
    current_pi_calibration,
    readout_due,
)
from zcu_lab.workflows.common.calibration_plots import (
    draw_frequency,
    draw_rabi,
    retain_frequency_fit,
    retain_trace,
)
from zcu_lab.workflows.common.calibration_stage_types import (
    CalibrationPoint,
    CalibrationPulses,
    FrequencyGuess,
    FrequencyStageResult,
    PointGuess,
    RabiGuess,
    RabiStageResult,
    ReadoutPulses,
)
from zcu_lab.workflows.common.calibration_tunables import CalibrationTunables
from zcu_lab.workflows.common.stages import StageEnv


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


def _retained_readout(pulses: CalibrationPulses, carry: CalCarry) -> PulseReadoutCfg:
    """Copy the plan's readout with the latest accepted (or seed) freq and gain."""
    prior = carry.seed if carry.last is None else carry.last
    readout = pulses.readout.model_copy(deep=True)
    readout.set_param("freq", prior.readout_freq_mhz)
    readout.set_param("gain", prior.readout_gain)
    return readout


def _prior(carry: CalCarry) -> Calibrated:
    return carry.seed if carry.last is None else carry.last


def _record_frequency(
    scope: Scope,
    env: StageEnv,
    point: CalibrationPoint,
    carry: CalCarry,
    stage: FrequencyStageResult,
) -> None:
    """Update accepted frequency history and redraw its heatmap from a stage result."""
    record = stage.record
    if record.status == "accepted":
        assert record.frequency_mhz is not None
        carry.last = replace(_prior(carry), f01_mhz=record.frequency_mhz)
        carry.f01_by_flux.append((point.flux, record.frequency_mhz))
    if stage.fit is None:
        return
    assert record.scan_center_mhz is not None
    retain_frequency_fit(
        carry.frequency_traces,
        stage.fit,
        point.index,
        point.flux,
        record.scan_center_mhz,
    )
    draw_frequency(
        env.plots,
        carry.frequency_traces,
        carry.f01_by_flux,
        flux_unit=point.flux_unit,
        directory=scope.dir,
    )


def _record_rabi(
    scope: Scope,
    env: StageEnv,
    point: CalibrationPoint,
    carry: CalCarry,
    stage: RabiStageResult,
) -> None:
    """Apply Len then Amp acceptance to the pi history and redraw both heatmaps."""
    if stage.len_trace is not None:
        retain_trace(carry.rabi_traces, stage.len_trace)
        draw_rabi(
            env.plots,
            carry.rabi_traces,
            flux_unit=point.flux_unit,
            directory=scope.dir,
        )
    length = stage.length_record
    if length is not None and length.status == "accepted":
        assert length.frequency_mhz is not None and length.pi_length_us is not None
        prior = _prior(carry)
        carry.last = Calibrated(
            length.frequency_mhz,
            length.pi_length_us,
            prior.readout_freq_mhz,
            prior.readout_gain,
            drive_gain=length.drive_gain,
        )
    if stage.amp_trace is not None:
        retain_trace(carry.amp_traces, stage.amp_trace)
        draw_rabi(
            env.plots,
            carry.amp_traces,
            flux_unit=point.flux_unit,
            directory=scope.dir,
            coordinate="gain",
        )
    amp = stage.record.amp
    if amp is not None and amp.status == "accepted":
        assert stage.record.frequency_mhz is not None
        carry.last_amp_gain = stage.record.drive_gain
        carry.last = replace(
            _prior(carry),
            f01_mhz=stage.record.frequency_mhz,
            pi_length_us=amp.length_us,
            drive_gain=stage.record.drive_gain,
        )


def _skipped_readout(reason: str) -> ReadoutResult:
    return ReadoutResult(
        status="skipped", reason=reason, frequency_mhz=None, gain=None, file=None
    )


def _readout_pulses(
    pulses: CalibrationPulses, carry: CalCarry, pi: Calibrated
) -> ReadoutPulses:
    """Prepare this point's accepted pi and the retained readout for the scan."""
    drive = pulses.qub.model_copy(deep=True)
    drive.set_param("freq", pi.f01_mhz)
    drive.set_param("length", pi.pi_length_us)
    if pi.drive_gain is not None:
        drive.set_param("gain", pi.drive_gain)
    return ReadoutPulses(pi=drive, readout=_retained_readout(pulses, carry))


def _record_readout(
    point: CalibrationPoint, carry: CalCarry, readout: ReadoutResult
) -> None:
    """Retain an accepted readout's freq/gain and its producer point."""
    if readout.status != "accepted":
        return
    assert readout.frequency_mhz is not None and readout.gain is not None
    carry.last = replace(
        _prior(carry),
        readout_freq_mhz=readout.frequency_mhz,
        readout_gain=readout.gain,
    )
    carry.readout_source = CalibrationSource(
        source_index=point.index, source_flux=point.flux, fallback=False
    )


async def calibrate_point(
    scope: Scope,
    env: StageEnv,
    pulses: CalibrationPulses,
    settings: Live[CalibrationTunables],
    *,
    guess: PointGuess,
    carry: CalCarry,
) -> PointCalibrationResult:
    """Calibrate this point's frequency, pi/pi2 and due readout in order.

    pulses are the plan's drive, probe and readout templates; each stage
    receives the readout retained in carry. settings supplies fresh policy
    before every stage. guess names the point and its positive history-based T1
    estimate in us. carry holds the accepted history and feedback controllers:
    stages receive only frequency_tracker and rabi_drive; this function updates
    last, f01_by_flux, traces, last_amp_gain and readout_source from their
    results, then redraws the shared heatmaps.
    Rabi delegates len_only, amp_only or len_then_amp to calibrate_rabi,
    including AmpRabi where selected. Readout runs on cadence with this point's
    pi, otherwise it returns skipped without postponing its cadence.
    Cancellation returns results already reached, with later fields None.
    Expected failures/rejections return normal results. Unexpected exceptions
    propagate without returning an aggregate: frequency success then a Rabi bug
    leaves frequency's saved h5 on disk, but not in the caller's point record.
    The result is created and accumulated locally, never passed in by a caller.
    This function never receives or changes the point record.
    """
    result = PointCalibrationResult()
    point = guess.point
    if scope.cancelled:
        return result
    tracker = carry.frequency_tracker
    if tracker is None:
        raise ValueError("Frequency calibration requires a captured physical predictor")
    frequency = await calibrate_frequency(
        scope,
        env,
        replace(pulses, readout=_retained_readout(pulses, carry)),
        settings.current,
        guess=FrequencyGuess(point=point, seed_flux=carry.seed_flux),
        tracker=tracker,
    )
    result.frequency = frequency.record
    _record_frequency(scope, env, point, carry, frequency)
    if scope.cancelled:
        return result
    tun = settings.current
    rabi = await calibrate_rabi(
        scope,
        env,
        replace(pulses, readout=_retained_readout(pulses, carry)),
        tun,
        guess=RabiGuess(
            point=point,
            frequency_mhz=(
                frequency.record.frequency_mhz
                if frequency.record.status == "accepted"
                else None
            ),
            relax_us=guess.relax_us,
            prior_pi_us=_prior(carry).pi_length_us,
            target_pi_us=tun.resolve_target_pi_length_us(carry.seed.pi_length_us),
            seed_product=carry.seed.pi_length_us * float(pulses.qub.gain),
            m_ratio=frequency.m_ratio,
            last_amp_gain=carry.last_amp_gain,
        ),
        rabi_drive=carry.rabi_drive,
    )
    result.rabi = rabi.record
    _record_rabi(scope, env, point, carry, rabi)
    if scope.cancelled:
        return result
    tun = settings.current
    if not readout_due(point.index, tun):
        result.readout = _skipped_readout("Readout flux-point cadence")
        return result
    pi = current_pi_calibration(rabi.record, carry)
    if pi is None:
        result.readout = _skipped_readout(
            "Requires this point's accepted pi pulse (NOW)"
        )
        return result
    result.readout = await calibrate_readout(
        scope,
        env,
        _readout_pulses(pulses, carry, pi),
        tun,
        point=point,
        guess_us=guess.relax_us,
    )
    _record_readout(point, carry, result.readout)
    return result
