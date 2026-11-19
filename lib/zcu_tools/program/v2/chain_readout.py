"""Reduce thermally reset 16-read shots to additive moment sums."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import Trigger

from zcu_tools.program.acquisition import ShotSumProcessor

from .macro.readout import ReadoutEmission
from .modules.readout import PulseReadout
from .readout_labels import resolve_readout_label, validate_declared_reads
from .readout_observation import (
    ReadoutObservation,
    observe_readout_paths,
    observe_readouts,
)
from .timeline import TimelineProgramV2


def thermal_chain_sums(prog: TimelineProgramV2, read: str) -> ShotSumProcessor:
    """Resolve one PulseReadout and reduce its 16-read shots to 108 columns.

    read is an emit tag or an untagged module name. All emissions of that
    owner must be 16 unrolled reads or one read in one fixed Repeat of 16,
    on every feasible setting path, including implicit empty branch cases.
    Other triggers on its ADC are forbidden. The compiler supplies the
    per-shot ADC count. Missing/ambiguous labels, non-PulseReadout owners and
    emitted/compiled
    counts or control scopes outside this contract raise ValueError with
    expected and actual values. Construction does not modify prog.

    The caller arranges consecutive reads with no intervening drive or extra
    wait, followed by the thermal relaxation delay. Pulse/timing content is
    not checked. Every read supplies means and adjacent-pair moments.

    The callable borrows ADC-keyed float64 (points, reps, 16, 2) IQ arrays,
    without mutation. IQ is ADC integral / rounded integration length, without
    offset subtraction. Other ADCs are ignored; a missing selected ADC raises
    KeyError, and a shape mismatch raises ValueError. Output is float64
    (points, 108), summed over shots, never averaged:

    * 0: shot count.
    * 1:33: I,Q of R1..R16, in read order.
    * 33:36: R1 I²,IQ,Q².
    * 36:96: II′,IQ′,QI′,QQ′ for (R1,R2)..(R15,R16), in pair order.
    * 96:108: II′,IQ′,QI′,QQ′ for (R1,R3),(R1,R4),(R1,R5).

    All cross components are preserved, including the middle adjacent pair.
    No analysis weights are applied by this processor.

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
        first_i, first_q = values[:, :, 0, 0], values[:, :, 0, 1]
        sums = np.empty((points, 108), dtype=np.float64)
        sums[:, 0] = reps
        sums[:, 1:33] = values.sum(axis=1).reshape(points, 32)
        sums[:, 33:36] = np.stack(
            [first_i**2, first_i * first_q, first_q**2], axis=-1
        ).sum(axis=1)
        adjacent = np.einsum(
            "prki,prkj->pkij", values[:, :, :-1, :], values[:, :, 1:, :]
        )
        sums[:, 36:96] = adjacent.reshape(points, 60)
        early = np.einsum("pri,prkj->pkij", values[:, :, 0, :], values[:, :, 2:5, :])
        sums[:, 96:108] = early.reshape(points, 12)
        return sums

    return process


def _resolve_block(prog: TimelineProgramV2, read: str, reads: int) -> int:
    observations = list(observe_readouts(prog))
    owner = resolve_readout_label(observations, read)
    if not isinstance(owner, PulseReadout):
        raise ValueError(f"label {read!r}: expected PulseReadout, actual MuxReadout")
    _validate_block(owner, observations, reads)
    for path in observe_readout_paths(prog):
        _validate_block(owner, list(path.observations), reads)
    ch = owner.cfg.ro_ch
    validate_declared_reads(prog, ch, reads)
    return ch


def _validate_block(
    owner: PulseReadout, observations: list[ReadoutObservation], reads: int
) -> None:
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
