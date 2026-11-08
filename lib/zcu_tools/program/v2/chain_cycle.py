"""Per-cycle augmented moment sums for sixteen ordered pulse reads."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import Trigger

from zcu_tools.program.acquisition import ShotSumProcessor

from .macro.readout import ReadoutEmission
from .modules.readout import PulseReadout
from .readout_observation import (
    ReadoutObservation,
    observe_readout_paths,
    observe_readouts,
)
from .timeline import TimelineProgramV2


def chain_cycle_sums(prog: TimelineProgramV2, read: str) -> ShotSumProcessor:
    """Resolve one pulse-readout owner and sum 65 cycle statistics over reps.

    prog is a built TimelineProgramV2. read is an emit tag, or the module name
    of an untagged PulseReadout. The label must resolve to one owner; all that
    owner's emissions belong to the cycle, including emissions with other tags.
    Multiple owners matching the label are ambiguous. The selected ADC must
    have 16 unrolled reads, or two source reads in a fixed Repeat of eight, and
    no other triggers. Call prog.set_reads_per_shot(16) after build and before
    this function. Construction does not change that declaration. Other ADCs
    are ignored.

    The caller supplies a thermalised (D R R)^8 shot: read 2k follows operation
    D_k, read 2k+1 follows a read. Only read counts, owner and control structure
    are checked here, not D placement or thermalisation. A different operation
    order gives the following sums a different meaning.

    The processor borrows ADC-keyed float64 (points, reps, 16, 2) IQ arrays,
    without mutation, and returns float64 (points, 66, 66) sums of z zᵀ over
    reps, where z = (1, s) and s has these zero-based slots:

    * 0:32: I,Q for each read in time order.
    * 32:41: I²,IQ,Q² summed over read 0; reads 2,4,...,14; reads 1,3,...,15.
    * 41:65: II,IQ,QI,QQ summed over ordered pairs i<j in six groups. With
      d = number of D operations between reads and g = j-i, the groups are
      (d=0,g=1), (d=1,g=1), (d=1,g=2..3), (d=2,g=2..3), (d=2,g>=4),
      (d>=3), containing 8,7,21,6,18,60 pairs respectively.

    Output [0,0] counts cycles, [0,1:] contains the 65 sums, and [1:,1:]
    contains their product sums. No within-group or rep normalization is
    applied; acquire_summed adds completed rounds.

    Missing/ambiguous labels, unsupported read/control structure, extra
    triggers, and declared or incoming read/IQ shape mismatches raise
    ValueError with expected and actual values. Missing ADC input raises
    KeyError. The program must remain unmodified after resolution.
    """
    ch = _resolve_cycle(prog, read)

    def process(iq: Mapping[int, NDArray[np.float64]]) -> NDArray[np.float64]:
        values = iq[ch]
        if values.ndim != 4 or values.shape[2:] != (16, 2):
            raise ValueError(
                f"ADC {ch}: expected (points, reps, 16, 2); actual {values.shape}"
            )
        vectors = np.zeros((*values.shape[:2], 66), dtype=np.float64)
        vectors[..., 0] = 1
        vectors[..., 1:33] = values.reshape(*values.shape[:2], 32)
        i, q = values[..., 0], values[..., 1]
        for group, reads in enumerate(([0], range(2, 16, 2), range(1, 16, 2))):
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
                start = 42 + 4 * _pair_group(left, right)
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

    return process


def _pair_group(left: int, right: int) -> int:
    d, gap = right // 2 - left // 2, right - left
    if d == 0:
        return 0
    if d == 1:
        return 1 if gap == 1 else 2
    if d == 2:
        return 3 if gap <= 3 else 4
    return 5


def _resolve_cycle(prog: TimelineProgramV2, read: str) -> int:
    observations = list(observe_readouts(prog))
    labels: dict[str, list[PulseReadout]] = {}
    for observation in observations:
        source = observation.source
        if isinstance(source, ReadoutEmission) and isinstance(
            source.owner, PulseReadout
        ):
            label = source.owner.name if source.label is None else source.label
            owners = labels.setdefault(label, [])
            if not any(owner is source.owner for owner in owners):
                owners.append(source.owner)
    matches = labels.get(read, [])
    if len(matches) != 1:
        raise ValueError(
            f"label {read!r}: expected one PulseReadout owner, actual {len(matches)}; "
            f"owners: {[owner.name for owner in matches]}; available labels: {list(labels)}"
        )
    _validate_read_structure(matches[0], observations)
    for path in observe_readout_paths(prog):
        _validate_read_structure(matches[0], list(path.observations))
    ch = matches[0].cfg.ro_ch
    declared = prog.ro_chs[ch]["trigs"]
    if declared != 16:
        raise ValueError(f"ADC {ch}: expected 16 reads per shot; actual {declared}")
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
