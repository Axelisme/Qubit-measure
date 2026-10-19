"""Shared tone settings and IQ data validation for mux Ramsey and Echo."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import ProgramV2Cfg, PulseCfg, SweepCfg

from .mux_sweep_cfg import (
    MuxDriveModuleCfg,
    MuxToneCfg,
    validate_mux_scan,
)


class MuxCoherenceToneCfg(ConfigBase):
    """One named original tone's required x90 drive and MHz detune.

    name is its nonempty capture-generated qubit identity. drive is the fixed
    pi/2 source pulse. detune ramps the final pi/2 phase, not its carrier.
    Full-list admission validates every pulse, detune and independent channel,
    including nonparticipants, and rejects duplicate names.
    """

    name: Annotated[str, FieldMeta(readonly=True)]
    drive: PulseCfg
    detune: float = 0.0


def validate_coherence_scan(
    cfg: ProgramV2Cfg,
    modules: MuxDriveModuleCfg,
    tones: Sequence[MuxCoherenceToneCfg],
    delay: SweepCfg,
    participants: Sequence[str],
) -> None:
    """Check the tones and shared delay before a Ramsey or Echo measurement.

    cfg and modules supply shared averaging, timing and the full mux readout.
    participants selects tone names in measurement order. Require finite
    nonnegative delay bounds in us, finite detunes in MHz, fixed drive pulses
    and independent channels. Raise ValueError without changing inputs.
    """
    # Check the shared evolution window and each final-pulse phase ramp.
    if not all(math.isfinite(v) for v in (delay.start, delay.stop, delay.step)):
        raise ValueError("delay start/stop/step must be finite")
    if min(delay.start, delay.stop) < 0:
        raise ValueError("delay must be nonnegative us")
    for index, tone in enumerate(tones):
        if not math.isfinite(tone.detune):
            raise ValueError(f"tones[{index}].detune must be finite MHz")

    # Reuse full-tone checks with the same delay grid for every drive.
    validate_mux_scan(
        cfg,
        modules,
        [MuxToneCfg(name=t.name, drive=t.drive, sweep=delay) for t in tones],
        participants,
    )


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
    source_frequencies: float64 (K,), MHz from each source x90 pulse before ramp.
    Full pulse cfg and inactive tones remain in RunRecord.cfg. Cancellation and
    acquisition error status belong to the RunContext StopSignal, not these data.
    """

    tone_indices: NDArray[np.int64]
    times: NDArray[np.float64]
    signals: NDArray[np.complex128]
    actual_detunes: NDArray[np.float64]
    source_frequencies: NDArray[np.float64]


def validate_coherence_result(result: MuxCoherenceResult) -> None:
    """Check Ramsey/Echo delay, IQ and per-tone frequency array shapes and dtypes.

    Require K,P>0, unique nonnegative indices and nonnegative finite requested
    times. Source frequencies must be finite. Actual detunes may be NaN when the
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
        raise ValueError("times must be finite nonnegative float64 (P,) us")

    # Partial acquisition may leave NaN IQ, but never infinite samples.
    if result.signals.dtype != np.dtype(np.complex128) or result.signals.shape != (
        indices.size,
        times.size,
    ):
        raise ValueError("signals must be complex128 (K,P)")
    if np.any(np.isinf(result.signals)):
        raise ValueError("signals may contain NaN partial data, not infinity")

    # An unavailable rounded detune stays NaN; source carriers must be known.
    detunes, frequencies = result.actual_detunes, result.source_frequencies
    if (
        detunes.dtype != np.dtype(np.float64)
        or detunes.shape != indices.shape
        or np.any(np.isinf(detunes))
    ):
        raise ValueError("actual_detunes must be float64 (K,), finite or NaN")
    if (
        frequencies.dtype != np.dtype(np.float64)
        or frequencies.shape != indices.shape
        or not np.all(np.isfinite(frequencies))
    ):
        raise ValueError("source_frequencies must be finite float64 (K,)")
