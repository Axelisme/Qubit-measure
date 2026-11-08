"""Compile a repeated readout into per-dwell augmented moment sums."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import Trigger

from zcu_tools.program.acquisition import ShotSumProcessor

from .macro.readout import ReadoutEmission
from .modules.readout import PulseReadout
from .readout_labels import (
    emission_label,
    resolve_readout_label,
    validate_declared_reads,
)
from .readout_observation import observe_readouts
from .timeline import TimelineProgramV2


def chain_sweep_sums(prog: TimelineProgramV2, read: str) -> ShotSumProcessor:
    """Resolve a compiled pulse readout label and sum chain statistics over reps.

    read is an emit tag, or the module name of an untagged PulseReadout. Its ADC
    supplies all reads of a shot in time order. The caller must declare the chain
    length with prog.set_reads_per_shot(n) after build and before this call.
    The selected ADC must have exactly one trigger inside that single fixed
    Repeat, with n >= 16, and no other triggers. Missing/ambiguous labels,
    unsupported readout/control structure, fewer than 16 reads, and declared or
    incoming read counts that differ from the Repeat raise ValueError with
    expected and actual values. Other ADC inputs are ignored.

    The returned processor borrows ADC-keyed float64 (points, reps, n, 2) arrays
    without mutation and returns float64 (points, 12, 12) augmented sums Σ z zᵀ.
    z contains 1, mean I/Q, mean I²/IQ/Q², lag-1 II/(IQ+QI)/QQ means, then the
    sums of those three lag means over lags 2–15. Lag k uses n-k pairs inside
    each shot, never across reps. Entry [0,0] counts dwells, [0,1:] holds the
    eleven sums and [1:,1:] holds their products. No normalization over reps or
    rounds is applied; acquire_summed adds completed rounds.
    """
    ch, n = _resolve_chain(prog, read)

    def process(iq: Mapping[int, NDArray[np.float64]]) -> NDArray[np.float64]:
        values = iq[ch]
        if values.ndim != 4 or values.shape[2:] != (n, 2):
            raise ValueError(
                f"ADC {ch}: expected (points, reps, {n}, 2); actual {values.shape}"
            )
        i, q = values[..., 0], values[..., 1]
        vectors = np.empty((*values.shape[:2], 12), dtype=np.float64)
        vectors[..., 0] = 1
        vectors[..., 1:6] = np.stack(
            [
                i.mean(axis=2),
                q.mean(axis=2),
                (i * i).mean(axis=2),
                (i * q).mean(axis=2),
                (q * q).mean(axis=2),
            ],
            axis=-1,
        )
        vectors[..., 6:] = 0
        for lag in range(1, 16):
            correlations = np.stack(
                [
                    (i[..., :-lag] * i[..., lag:]).mean(axis=2),
                    (i[..., :-lag] * q[..., lag:] + q[..., :-lag] * i[..., lag:]).mean(
                        axis=2
                    ),
                    (q[..., :-lag] * q[..., lag:]).mean(axis=2),
                ],
                axis=-1,
            )
            start = 6 if lag == 1 else 9
            vectors[..., start : start + 3] += correlations
        return np.einsum("pri,prj->pij", vectors, vectors)

    return process


def _resolve_chain(prog: TimelineProgramV2, read: str) -> tuple[int, int]:
    observations = list(observe_readouts(prog))
    owner = resolve_readout_label(observations, read)
    if not isinstance(owner, PulseReadout):
        raise ValueError(f"label {read!r}: expected PulseReadout, actual MuxReadout")
    matches = [
        item
        for item in observations
        if isinstance(item.source, ReadoutEmission)
        and item.source.owner is owner
        and emission_label(item.source) == read
    ]
    if len(matches) != 1:
        raise ValueError(
            f"label {read!r}: expected one PulseReadout emission, actual {len(matches)}"
        )
    repeats = matches[0].repeat_counts
    ch = owner.cfg.ro_ch
    if len(repeats) != 1 or not isinstance(repeats[0], int):
        raise ValueError(
            f"label {read!r}: expected one fixed Repeat; actual repeat counts {repeats}"
        )
    n = repeats[0]
    if n < 16:
        raise ValueError(f"ADC {ch}: expected at least 16 reads per shot; actual {n}")
    triggers = [
        item
        for item in observations
        if isinstance(item.source, Trigger) and ch in item.source.ros
    ]
    if len(triggers) != 1:
        raise ValueError(
            f"ADC {ch}: expected one source trigger in the chain; actual {len(triggers)}"
        )
    if triggers[0].control_scope != "Repeat" or triggers[0].repeat_counts != repeats:
        raise ValueError(
            f"ADC {ch}: expected trigger inside Repeat {repeats}; actual "
            f"{triggers[0].control_scope} {triggers[0].repeat_counts}"
        )
    validate_declared_reads(prog, ch, n)
    return ch, n
