"""Shared tone settings and IQ data validation for mux Ramsey and Echo."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.program.v2 import Pulse, SweepCfg

from .mux_sweep_cfg import MuxAcquisition, MuxDriveToneCfg, validate_mux_tones


class MuxCoherenceToneCfg(MuxDriveToneCfg):
    """One named original tone's nullable x90 drive and MHz detune.

    name is its nonempty capture-generated qubit identity. drive is the fixed
    pi/2 source pulse, required only for participants and otherwise nullable.
    detune ramps the final pi/2 phase, not its carrier.
    Evolution uses the cfg's shared delay; no per-tone sweep is required.
    Admission validates participant pulses, detunes and independent channels,
    and rejects duplicate names in the complete list.
    """

    drive: Annotated[
        Pulse.Cfg | None,
        FieldMeta(
            label="π/2 pulse",
            description="Enter the calibrated x90 pulse used at both ends of the sequence; nonparticipants may leave it blank.",
        ),
    ] = None
    detune: Annotated[
        float,
        FieldMeta(
            label="Applied phase-ramp detune",
            unit="MHz",
            description="Enter the signed final π/2 phase slope versus total delay, not the RF carrier or the quantized slope.",
        ),
    ] = 0.0


def validate_coherence_scan(
    cfg: MuxAcquisition,
    tones: Sequence[MuxCoherenceToneCfg],
    delay: SweepCfg,
    participants: Sequence[int],
) -> None:
    """Check the tones and shared delay before a Ramsey or Echo measurement.

    cfg and modules supply shared averaging, timing and the full mux readout.
    participants selects original indices in measurement order. Require finite
    nonnegative delay bounds in us, finite detunes in MHz, fixed drive pulses
    and independent channels. Raise ValueError without changing inputs.
    """
    # Validate original indices and fixed drives before indexing any tone.
    validate_mux_tones(cfg, tones, participants)

    # Check the shared evolution window and each final-pulse phase ramp.
    if min(delay.start, delay.stop) < 0:
        raise ValueError("delay must be nonnegative µs")
    for index in participants:
        tone = tones[index]
        if not math.isfinite(tone.detune):
            raise ValueError(f"tones[{index}].detune must be finite MHz")


@dataclass(frozen=True)
class MuxCoherenceResult:
    """Explicit Ramsey/Echo source in original active tone order.

    tone_indices: int64 (K,), original full readout indices, unique/nonnegative.
    times: float64 (P,), common requested total free-evolution delay in us, not
    compiled actual delay. Echo splits this into two native half-delay waits.
    signals: complex128 (K,P), ADC-length-normalized average I+jQ, NaN if unwritten.
    actual_detunes: float64 (K,), MHz phase-ramp rates from actual_detune owner.
    May contain NaN when nonzero detune has no positive rounded delay; preserve
    the owner's diagnostics, never substitute zero/nominal detune.
    Source x90 carriers and inactive tones remain in RunRecord.cfg. Cancellation and
    acquisition error status belong to the RunContext StopSignal, not these data.
    iq_moments: optional float64 (K,P,6) aligned with signals, in columns
    [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²] after ADC offset correction. Run and Native
    carry these completed-round sums when available. Older Native files without
    this variable restore None. Labber exports the mean IQ without the sums.
    """

    tone_indices: NDArray[np.int64]
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    actual_detunes: NDArray[np.float64]
    iq_moments: NDArray[np.float64] | None = None


def validate_coherence_result(result: MuxCoherenceResult) -> None:
    """Check Ramsey/Echo delay, IQ and applied detune array shapes and dtypes.

    Require K,P>0, unique nonnegative indices and nonnegative finite requested
    times. Actual detunes may be NaN when the
    owner cannot estimate a rate, but not infinite. Raises ValueError, never fits or
    changes data; callers must not turn source-contract errors into fit failures.
    """
    # Keep original readout indices paired with the requested delay grid.
    indices, times = result.tone_indices, result.times
    if (
        indices.dtype != np.dtype(np.int64)
        or indices.ndim != 1
        or not indices.size
        or np.any(indices < 0)
        or np.unique(indices).size != indices.size
    ):
        raise ValueError("tone_indices must be unique nonnegative int64 (K,)")
    if (
        times.dtype != np.dtype(np.float64)
        or times.ndim != 1
        or not times.size
        or not np.all(np.isfinite(times))
        or np.any(times < 0)
    ):
        raise ValueError("times must be finite nonnegative float64 (P,) µs")

    # Partial acquisition may leave NaN IQ, but never infinite samples.
    if result.signals.dtype != np.dtype(np.complex128) or result.signals.shape != (
        indices.size,
        times.size,
    ):
        raise ValueError("signals must be complex128 (K,P)")
    if np.any(np.isinf(result.signals)):
        raise ValueError("signals may contain NaN partial data, not infinity")

    # An unavailable rounded detune stays NaN.
    detunes = result.actual_detunes
    if (
        detunes.dtype != np.dtype(np.float64)
        or detunes.shape != indices.shape
        or np.any(np.isinf(detunes))
    ):
        raise ValueError("actual_detunes must be float64 (K,), finite or NaN")
