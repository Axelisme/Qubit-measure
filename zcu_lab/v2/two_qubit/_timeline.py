"""Reset and tagged mux preamble shared by ordinary and heralded sequences."""

from zcu_tools.program.v2 import Reset, TimelineBuilder
from zcu_tools.program.v2.modules import MuxReadout


def emit_readout_prefix(
    builder: TimelineBuilder,
    readout_cfg: MuxReadout.Cfg,
    reset_cfg: Reset.Cfg | None,
    *,
    heralded: bool,
) -> MuxReadout:
    """Emit optional reset then optional herald; return this build's mux module.

    builder must be a fresh, unbuilt timeline. A fresh MuxReadout is created on
    every call. Emit the returned object again with tag='terminal' after the
    experiment sequence, so both reads share one ADC/resource declaration.
    Readout post_delay supplies the gap after herald; no extra delay is added.
    reset_cfg=None emits nothing. Caller builds and acquires separately.
    """
    readout = MuxReadout("readout", readout_cfg)
    if reset_cfg is not None:
        builder.emit(Reset("reset", reset_cfg))
    if heralded:
        builder.emit(readout, tag="herald")
    return readout
