"""Reduce one compiled mux readout to separate corrected IQ moments per tone."""

import numpy as np
from numpy.typing import NDArray
from qick.asm_v2 import Trigger

from zcu_tools.program.acquisition import ShotReducer
from zcu_tools.program.shot_processing import AdditiveShotReducer, ShotIQ

from .modules.readout import MuxReadout
from .readout_labels import resolve_readout_label
from .readout_observation import observe_readouts
from .timeline import TimelineProgramV2


def mux_iq_sums(prog: TimelineProgramV2, read: str) -> ShotReducer[NDArray[np.float64]]:
    """Build a reducer for the active tones of a compiled MuxReadout.

    read is the emit tag or untagged module name. Select the participating
    tones in the MuxReadout cfg before building prog; rows follow that cfg's
    tone order, not ADC channel or declaration order.

    Pass the reducer to prog.acquire_summed. Every completed round contributes
    float64 [N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²] per point and tone. The result and round
    hooks have shape (*point_dims, tones, 6). I and Q are divided by each
    ADC's integration length, then both are corrected by its native QICK
    offset before accumulation (ADR-0074). Static PFB registers are read
    while consuming the round; missing configuration raises RuntimeError.

    Require one unconditional integrated read per shot on each active ADC and
    no other ADCs, edge counting or shot masks. Invalid labels, programs and
    incoming read shapes raise ValueError. No cross-tone products are formed.
    """
    if prog.has_shot_mask_sources:
        raise ValueError("mux IQ sums do not support shot masks")
    channels = _resolve_mux_channels(prog, read)

    def round_sums(iq: ShotIQ, mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        if prog.has_shot_mask_sources or not mask.all():
            raise ValueError("mux IQ sums do not support shot masks")
        return np.stack([_tone_sums(prog, ch, iq[ch]) for ch in channels], axis=1)

    return AdditiveShotReducer(round_sums, (len(channels), 6))


def _resolve_mux_channels(prog: TimelineProgramV2, read: str) -> tuple[int, ...]:
    observations = list(observe_readouts(prog))
    owner = resolve_readout_label(observations, read)
    if not isinstance(owner, MuxReadout):
        raise ValueError(f"label {read!r}: expected MuxReadout")
    channels = tuple(tone.ro_ch for tone in owner.cfg.tones)
    if set(channels) != set(prog.ro_chs):
        raise ValueError(f"label {read!r}: only the active tone ADCs may be declared")
    for ch in channels:
        ro = prog.ro_chs[ch]
        if ro["edge_counting"] or ro["trigs"] != 1:
            raise ValueError(
                f"ADC {ch}: mux IQ sums require one integrated read per shot, not edge counting"
            )
        trigger_scopes = [
            obs.control_scope
            for obs in observations
            if isinstance(obs.source, Trigger) and ch in obs.source.ros
        ]
        if trigger_scopes != [None]:
            raise ValueError(
                f"ADC {ch}: expected one unconditional source trigger; found {trigger_scopes}"
            )
    return channels


def _tone_sums(
    prog: TimelineProgramV2, ch: int, shots: NDArray[np.float64]
) -> NDArray[np.float64]:
    if shots.ndim != 4 or shots.shape[2:] != (1, 2):
        raise ValueError(f"ADC {ch}: expected (points, reps, 1, 2); got {shots.shape}")
    iq = shots[:, :, 0] - prog.native_readout_iq_offset(ch)
    i, q = iq[..., 0], iq[..., 1]
    points, reps = i.shape
    return np.stack(
        [
            np.full(points, reps, dtype=np.float64),
            i.sum(axis=1),
            q.sum(axis=1),
            (i * i).sum(axis=1),
            (i * q).sum(axis=1),
            (q * q).sum(axis=1),
        ],
        axis=-1,
    )
