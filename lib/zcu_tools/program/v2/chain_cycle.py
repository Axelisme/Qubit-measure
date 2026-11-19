"""Per-cycle augmented moment sums for sixteen ordered pulse reads."""

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import Trigger

from zcu_tools.core.chain_cycle_groups import pair_group, same_read_group
from zcu_tools.program.acquisition import ShotReducer
from zcu_tools.program.shot_processing import (
    AdditiveShotReducer,
    ShotIQ,
    zero_rejected_iq,
)

from .macro.readout import ReadoutEmission
from .modules.readout import PulseReadout
from .readout_labels import resolve_readout_label, validate_declared_reads
from .readout_observation import (
    ReadoutObservation,
    observe_readout_paths,
    observe_readouts,
)
from .timeline import TimelineProgramV2


def chain_cycle_sums(
    prog: TimelineProgramV2, read: str
) -> ShotReducer[NDArray[np.float64]]:
    """Resolve one pulse-readout owner and sum 65 cycle statistics over reps.

    prog is a built TimelineProgramV2. read is an emit tag, or the module name
    of an untagged PulseReadout. The label must resolve to one owner; all that
    owner's emissions belong to the cycle, including emissions with other tags.
    Multiple readout owners matching the label, pulse or mux, are ambiguous; a
    label whose one owner is a MuxReadout is rejected. The selected ADC must
    have 16 unrolled reads, or two source reads in a fixed Repeat of eight, and
    no other triggers. The compiler supplies the per-shot ADC count. Other
    ADCs are ignored.

    The caller supplies a thermalised (D R R)^8 shot: read 2k follows operation
    D_k, read 2k+1 follows a read. Only read counts, owner and control structure
    are checked here, not D placement or thermalisation. A different operation
    order gives the following sums a different meaning.

    The reducer borrows ADC-keyed float64 (points, reps, 16, 2) IQ arrays.
    It accumulates float64 (points, 66, 66) sums of z zᵀ over kept reps,
    where z = (1, s) and s has these zero-based slots:

    * 0:32: I,Q for each read in time order.
    * 32:41: I²,IQ,Q² summed over read 0; reads 2,4,...,14; reads 1,3,...,15.
    * 41:65: II,IQ,QI,QQ summed over ordered pairs i<j in six groups. With
      d = number of D operations between reads and g = j-i, the groups are
      (d=0,g=1), (d=1,g=1), (d=1,g=2..3), (d=2,g=2..3), (d=2,g>=4),
      (d>=3), containing 8,7,21,6,18,60 pairs respectively.

    Output [0,0] counts cycles, [0,1:] contains the 65 sums, and [1:,1:]
    contains their product sums. No within-group or rep normalization is
    applied. The reducer adds completed rounds.

    Missing/ambiguous labels, unsupported read/control structure, extra
    triggers, and compiled read-count or incoming IQ shape mismatches raise
    ValueError with expected and actual values. Missing ADC input raises
    KeyError. The program must remain unmodified after resolution.
    """
    ch = _resolve_cycle(prog, read)

    width = 66

    def round_sums(iq: ShotIQ, mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        values = iq[ch]
        if values.ndim != 4 or values.shape[2:] != (16, 2):
            raise ValueError(
                f"ADC {ch}: expected (points, reps, 16, 2); actual {values.shape}"
            )
        values = zero_rejected_iq(values, mask)
        vectors = np.zeros((*values.shape[:2], width), dtype=np.float64)
        vectors[..., 0] = mask
        vectors[..., 1:33] = values.reshape(*values.shape[:2], 32)
        i, q = values[..., 0], values[..., 1]
        for group in range(3):
            reads = [read for read in range(16) if same_read_group(read) == group]
            vectors[..., 33 + 3 * group : 36 + 3 * group] = np.stack(
                [
                    (i[..., reads] ** 2).sum(axis=-1),
                    (i[..., reads] * q[..., reads]).sum(axis=-1),
                    (q[..., reads] ** 2).sum(axis=-1),
                ],
                axis=-1,
            )
        for left in range(16):
            for right in range(left + 1, 16):
                start = 42 + 4 * pair_group(left, right)
                vectors[..., start : start + 4] += np.stack(
                    [
                        i[..., left] * i[..., right],
                        i[..., left] * q[..., right],
                        q[..., left] * i[..., right],
                        q[..., left] * q[..., right],
                    ],
                    axis=-1,
                )
        return np.einsum("pri,prj->pij", vectors, vectors)

    return AdditiveShotReducer(round_sums, (width, width))


def _resolve_cycle(prog: TimelineProgramV2, read: str) -> int:
    observations = list(observe_readouts(prog))
    owner = resolve_readout_label(observations, read)
    if not isinstance(owner, PulseReadout):
        raise ValueError(f"label {read!r}: expected PulseReadout, actual MuxReadout")
    _validate_read_structure(owner, observations)
    for path in observe_readout_paths(prog):
        _validate_read_structure(owner, list(path.observations))
    ch = owner.cfg.ro_ch
    validate_declared_reads(prog, ch, 16)
    return ch


def _validate_read_structure(
    owner: PulseReadout, observations: list[ReadoutObservation]
) -> None:
    reads = [
        item
        for item in observations
        if isinstance(item.source, ReadoutEmission) and item.source.owner is owner
    ]
    repeats = [item.repeat_counts for item in reads]
    if repeats not in [[()] * 16, [(8,)] * 2]:
        raise ValueError(
            f"ADC {owner.cfg.ro_ch}: expected 16 unrolled reads or 2 reads in "
            f"fixed Repeat(8); actual {len(reads)} source reads, repeat counts {repeats}"
        )
    triggers = [
        item
        for item in observations
        if isinstance(item.source, Trigger) and owner.cfg.ro_ch in item.source.ros
    ]
    if len(triggers) != len(reads):
        raise ValueError(
            f"ADC {owner.cfg.ro_ch}: expected {len(reads)} source triggers from "
            f"the selected owner only; actual {len(triggers)}"
        )
    expected_scope = "Repeat" if repeats[0] else None
    actual = [(item.control_scope, item.repeat_counts) for item in triggers]
    expected = [(expected_scope, repeat) for repeat in repeats]
    if actual != expected:
        raise ValueError(
            f"ADC {owner.cfg.ro_ch}: expected trigger control {expected}; actual {actual}"
        )
