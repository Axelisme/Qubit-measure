"""Stationary R-pi readout sums over complete acquisition rounds."""

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import Trigger

from zcu_tools.program._shot_processing import (
    AdditiveShotReducer,
    ShotIQ,
    zero_rejected_iq,
)
from zcu_tools.program.acquisition import ShotReducer

from .macro.readout import ReadoutEmission
from .modules.readout import PulseReadout
from .readout_labels import resolve_readout_label, validate_declared_reads
from .readout_observation import observe_readout_paths, observe_readouts
from .timeline import TimelineProgramV2


def rpi_sums(prog: TimelineProgramV2, read: str) -> ShotReducer[NDArray[np.float64]]:
    """Resolve a compiled R-pi readout and reduce a complete round to sums.

    read is an emit tag or untagged PulseReadout name. The caller owns the Rπ
    schedule and calibrated π, including fixed π settings and readout post_delay
    measured from the generator end. Use relax_delay=0 to avoid added waiting
    between reps. Multiple settings require compiled reps as the innermost loop
    (reps_innermost=True); a single setting allows either loop order. Invalid
    compiled loop order raises ValueError at construction.
    The label must select one PulseReadout emission inside one fixed Repeat
    of at least 16, with no other triggers on its ADC or conditional emission.
    After build call prog.set_reads_per_shot(n) with the Repeat count, before
    constructing the reducer. Label/provenance/trigger/count mismatches raise
    ValueError with expected and actual values, without changing prog.
    No pulse physics is checked. Discontinuities between reps do not affect
    lags; hidden discontinuities inside a rep cycle cannot be detected.

    Each round borrows ADC float64 (points, reps, reads, 2) IQ for one complete
    round: integer integrals / rounded integration length, without IQ-offset
    subtraction. Each rep at each setting is one segment of length reads.
    Lag 1..15 pairs stay within that cycle. The reducer accumulates sums
    across kept reps and completed rounds. Rep, round and setting boundaries never contribute pairs. No inputs
    are changed and other ADC inputs are ignored. Shape/dtype mismatches raise
    ValueError with expected and actual values; a missing selected ADC raises
    KeyError. Points, reps and reads must match the compiled program.

    Return float64 (points, 35) additive records, with Python column indices:
    0: kept cycle count S; 1: read count N = S * reads;
    2:4: ΣI, ΣQ; 4: Σ(I²+Q²); 5:20: lag dot-product sums j=1..15;
    20:35: actual integer pair counts C_j. These are 18 moment sums, in ADC
    units for first moments and ADC units squared for square/products. Counts
    are integers encoded as float64. H=1 metadata recovers segment length L=N/S,
    not N itself; do not combine records with different L. Each kept segment adds
    max(L-j, 0) pairs. Pass one
    pooled record to RPiSums.from_processor in
    analysis.readout_chain; no per-round normalization is applied here.
    """
    dims, avg = prog.loop_dims, prog.avg_level
    if dims is None or avg is None:
        raise ValueError("R-pi sums require compiled acquisition loops")
    points = int(np.prod([size for index, size in enumerate(dims) if index != avg]))
    if points > 1 and avg != len(dims) - 1:
        raise ValueError("R-pi multiple settings require reps_innermost=True")
    ch, reads = _resolve_readout(prog, read)
    expected_shape = (points, dims[avg], reads, 2)

    def round_sums(iq: ShotIQ, mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        values = iq[ch]
        if values.shape != expected_shape or values.dtype != np.float64:
            raise ValueError(
                f"ADC {ch}: expected float64 {expected_shape}; actual "
                f"{values.dtype} {values.shape}"
            )
        values = zero_rejected_iq(values, mask)
        cycles = mask.sum(axis=1)
        sums = np.zeros((points, 35), dtype=np.float64)
        sums[:, 0] = cycles
        sums[:, 1] = cycles * reads
        sums[:, 2:4] = values.sum(axis=(1, 2))
        sums[:, 4] = np.einsum("prti,prti->p", values, values)
        for lag in range(1, 16):
            early = values[:, :, :-lag]
            late = values[:, :, lag:]
            sums[:, 4 + lag] = np.einsum("prti,prti->p", early, late)
            sums[:, 19 + lag] = cycles * max(reads - lag, 0)
        return sums

    return AdditiveShotReducer(round_sums)


def _resolve_readout(prog: TimelineProgramV2, read: str) -> tuple[int, int]:
    observations = list(observe_readouts(prog))
    owner = resolve_readout_label(observations, read)
    if not isinstance(owner, PulseReadout):
        raise ValueError(f"label {read!r}: expected PulseReadout, actual MuxReadout")
    emissions = [
        item
        for item in observations
        if isinstance(item.source, ReadoutEmission) and item.source.owner is owner
    ]
    if len(emissions) != 1:
        raise ValueError(
            f"label {read!r}: expected one emission, actual {len(emissions)}"
        )
    emission = emissions[0]
    repeats = emission.repeat_counts
    ch = owner.cfg.ro_ch
    if len(repeats) != 1 or not isinstance(repeats[0], int) or repeats[0] < 16:
        raise ValueError(
            f"ADC {ch}: expected one fixed Repeat of at least 16; actual {repeats}"
        )
    if isinstance(emission.source, ReadoutEmission) and emission.source.ro_ch != ch:
        raise ValueError(
            f"label {read!r}: expected ADC {ch}; actual ADC {emission.source.ro_ch}"
        )
    triggers = [
        item
        for item in observations
        if isinstance(item.source, Trigger) and ch in item.source.ros
    ]
    if len(triggers) != 1:
        raise ValueError(
            f"ADC {ch}: expected one source trigger; actual {len(triggers)}"
        )
    trigger = triggers[0]
    if trigger.control_scope != "Repeat" or trigger.repeat_counts != repeats:
        raise ValueError(
            f"ADC {ch}: expected trigger in Repeat {repeats}; actual "
            f"{trigger.control_scope} {trigger.repeat_counts}"
        )
    for path in observe_readout_paths(prog):
        if not any(item.source is trigger.source for item in path.observations):
            raise ValueError(
                f"ADC {ch}: expected unconditional trigger; actual conditional"
            )
    validate_declared_reads(prog, ch, repeats[0])
    return ch, repeats[0]
