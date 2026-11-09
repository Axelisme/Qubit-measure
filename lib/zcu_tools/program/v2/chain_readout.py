"""Compile consecutive readout blocks into additive moment sums."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import Trigger

from zcu_tools.program.acquisition import ShotSumProcessor

from .macro.readout import ReadoutEmission
from .modules.readout import PulseReadout
from .readout_labels import resolve_readout_label, validate_declared_reads
from .readout_observation import observe_readouts
from .timeline import TimelineProgramV2


def chain_readout_sums(prog: TimelineProgramV2, read: str) -> ShotSumProcessor:
    """Resolve a compiled PulseReadout block and sum 113 moments over reps.

    read is an emit tag or the module name of an untagged PulseReadout. It must
    identify one pulse or mux readout owner, and that owner must be a
    PulseReadout; all emissions of that owner supply the shot's reads in time
    order, including emissions with other tags. The program must contain
    32 unrolled emissions or one emission inside one fixed Repeat of 32, with
    no other triggers on its ADC. After build, call prog.set_reads_per_shot(32)
    before constructing this processor. Construction does not change prog.

    The caller must arrange consecutive reads with no intervening drive or
    extra waiting. Only readout provenance, trigger structure and counts are
    checked, not pulse/timing content. A drive inside the block changes the
    meaning of the returned moments and is not detected here.

    The returned processor borrows ADC-keyed float64 (points, reps, 32, 2)
    IQ arrays without mutation; other ADCs are ignored. IQ is ADC integral /
    rounded integration length, without offset subtraction. It returns float64
    (points, 114): column 0 counts blocks, followed by sums of per-block
    moments in this order (column ranges below use Python slices):

    * 1:3: mean I, Q; 3:6: same-read I², IQ, Q².
    * 6:66: lag k=1..15, each with II′, IQ′, QI′, QQ′.
    * 66:90: lag k=1..4, each with I²I′, I²Q′, IQI′, IQQ′, Q²I′, Q²Q′.
    * 90:114: lag k=1..4, each with II′², II′Q′, IQ′², QI′², QI′Q′, QQ′².

    Primes denote the read k steps later. Each mean uses 32 reads; each lag
    moment uses its own 32-k pairs within the block, never across reps. Reps
    and rounds are not averaged: acquire_summed adds completed round sums.
    Missing/ambiguous labels, a MuxReadout owner, unsupported observed
    trigger/Repeat structure, and emitted, declared or incoming
    read-count/shape mismatches raise ValueError with expected and actual
    values. A missing selected ADC input raises KeyError.
    """
    ch = _resolve_block(prog, read)

    def process(iq: Mapping[int, NDArray[np.float64]]) -> NDArray[np.float64]:
        values = iq[ch]
        if values.ndim != 4 or values.shape[2:] != (32, 2):
            raise ValueError(
                f"ADC {ch}: expected (points, reps, 32, 2); actual {values.shape}"
            )
        points, reps = values.shape[:2]
        i, q = values[..., 0], values[..., 1]
        second = np.stack([i * i, i * q, q * q], axis=-1)
        sums = np.empty((points, 114), dtype=np.float64)
        sums[:, 0] = reps
        sums[:, 1:3] = values.mean(axis=2).sum(axis=1)
        sums[:, 3:6] = second.mean(axis=2).sum(axis=1)
        for lag in range(1, 16):
            cross = np.einsum(
                "prti,prtj->pij", values[..., :-lag, :], values[..., lag:, :]
            ) / (32 - lag)
            start = 6 + 4 * (lag - 1)
            sums[:, start : start + 4] = cross.reshape(points, 4)
        for lag in range(1, 5):
            early_pair = np.einsum(
                "prti,prtj->pij", second[..., :-lag, :], values[..., lag:, :]
            ) / (32 - lag)
            late_pair = np.einsum(
                "prti,prtj->pij", values[..., :-lag, :], second[..., lag:, :]
            ) / (32 - lag)
            start = 66 + 6 * (lag - 1)
            sums[:, start : start + 6] = early_pair.reshape(points, 6)
            sums[:, start + 24 : start + 30] = late_pair.reshape(points, 6)
        return sums

    return process


def thermal_chain_sums(prog: TimelineProgramV2, read: str) -> ShotSumProcessor:
    """Resolve one PulseReadout and reduce its 16-read shots to 38 columns.

    read is an emit tag or an untagged module name. All emissions of that
    owner must be 16 unrolled reads or one read in one fixed Repeat of 16.
    Other triggers on its ADC are forbidden. Build prog, then call
    prog.set_reads_per_shot(16) before constructing this processor.
    Missing/ambiguous labels, non-PulseReadout owners and emitted/declared
    counts or control scopes outside this contract raise ValueError with
    expected and actual values. Construction does not modify prog.

    The caller arranges consecutive reads with no intervening drive or extra
    wait, followed by the thermal relaxation delay. Pulse/timing content is
    not checked. Reads 6..14 evolve the state but supply no moments.

    The callable borrows ADC-keyed float64 (points, reps, 16, 2) IQ arrays,
    without mutation. IQ is ADC integral / rounded integration length, without
    offset subtraction. Other ADCs are ignored; a missing selected ADC raises
    KeyError, and a shape mismatch raises ValueError. Output is float64
    (points, 38), summed over shots, never averaged:

    * 0: shot count.
    * 1:15: I,Q of R1,R2,R3,R4,R5,R15,R16, in that order.
    * 15:18: R1 I²,IQ,Q².
    * 18:38: II′,IQ′,QI′,QQ′ for (R1,R2),(R1,R3),(R1,R4),
      (R1,R5),(R15,R16), in that order.

    Reps and completed rounds add, including the shot count.
    """
    ch = _resolve_block(prog, read, reads=16)

    def process(iq: Mapping[int, NDArray[np.float64]]) -> NDArray[np.float64]:
        values = iq[ch]
        if values.ndim != 4 or values.shape[2:] != (16, 2):
            raise ValueError(
                f"ADC {ch}: expected (points, reps, 16, 2); actual {values.shape}"
            )
        points, reps = values.shape[:2]
        recorded = values[:, :, [0, 1, 2, 3, 4, 14, 15], :]
        first_i, first_q = values[:, :, 0, 0], values[:, :, 0, 1]
        sums = np.empty((points, 38), dtype=np.float64)
        sums[:, 0] = reps
        sums[:, 1:15] = recorded.sum(axis=1).reshape(points, 14)
        sums[:, 15:18] = np.stack(
            [first_i**2, first_i * first_q, first_q**2], axis=-1
        ).sum(axis=1)
        cross = np.einsum(
            "prki,prkj->pkij",
            values[:, :, [0, 0, 0, 0, 14], :],
            values[:, :, [1, 2, 3, 4, 15], :],
        )
        sums[:, 18:38] = cross.reshape(points, 20)
        return sums

    return process


def _resolve_block(prog: TimelineProgramV2, read: str, reads: int = 32) -> int:
    observations = list(observe_readouts(prog))
    owner = resolve_readout_label(observations, read)
    if not isinstance(owner, PulseReadout):
        raise ValueError(f"label {read!r}: expected PulseReadout, actual MuxReadout")
    repeats = [
        observation.repeat_counts
        for observation in observations
        if isinstance(source := observation.source, ReadoutEmission)
        and source.owner is owner
    ]
    ch = owner.cfg.ro_ch
    if repeats != [(reads,)] and repeats != [()] * reads:
        raise ValueError(
            f"ADC {ch}: expected {reads} reads unrolled or one fixed Repeat of {reads}; "
            f"actual emission repeat counts {repeats}"
        )
    triggers = [
        item
        for item in observations
        if isinstance(item.source, Trigger) and ch in item.source.ros
    ]
    if len(triggers) != len(repeats):
        raise ValueError(
            f"ADC {ch}: expected {len(repeats)} source triggers for the block; "
            f"actual {len(triggers)}"
        )
    expected_scope = "Repeat" if repeats == [(reads,)] else None
    for trigger in triggers:
        if (
            trigger.control_scope != expected_scope
            or trigger.repeat_counts != repeats[0]
        ):
            raise ValueError(
                f"ADC {ch}: expected {'uncontrolled' if expected_scope is None else expected_scope} "
                f"triggers with repeat counts {repeats[0]}; actual "
                f"{trigger.control_scope} {trigger.repeat_counts}"
            )
    validate_declared_reads(prog, ch, reads)
    return ch
