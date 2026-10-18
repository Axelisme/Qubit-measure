"""Host zip acquisition shared by mux spectroscopy and amplitude Rabi."""

from collections.abc import Sequence
from functools import partial

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules import Join, MuxReadout, Pulse

from .mux import MuxPulseScan, MuxSweepResult
from .mux_sweep_cfg import MuxDriveModuleCfg


def _averaged_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    # Builder copies modules; collect only through the initialized program owner.
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_iq(prog, acquired, read_index=0)
            return iq[:, 0] + 1j * iq[:, 1]
    raise ValueError("Mux scan program is missing its named MuxReadout")


def acquire_mux_scan(
    cfg: ProgramV2Cfg,
    modules: MuxDriveModuleCfg,
    scan: MuxPulseScan,
    *,
    context: RunContext,
    coordinate_label: str,
) -> MuxSweepResult:
    """Acquire a prepared host zip scan with independent drives and common readout.

    cfg supplies shared reps/rounds and initial/relax delays (us). modules holds
    reset and the full ordered readout. scan must come from the leaf's validated
    preparation: original active indices, coordinates (K,P) in the label's units,
    and nominal point_pulses (P,K). Inputs are read-only. Every point compiles
    once as reset, Join of active pulses, then active mux readout. Join starts
    branches together; shorter branches wait without end-alignment padding.
    Native pulse timing and parameter conversion remain program-owned.

    context supplies compatible soc/soccfg, caller-owned Plots and a fresh shared
    StopSignal. Publish measurement_i magnitude plots for original tone indices.
    Return complex128 ADC-length-normalized average IQ (K,P), with each drive's
    own coordinates. The program averages reps/rounds; completed-round hooks
    preserve a partially averaged point on cancellation. Unwritten points remain
    NaN, including cancellation before any data. No files or records are created.
    Schedule retains acquisition errors on context.cancel_signal; consumers must
    call raise_if_error() before reporting success. Build/admission errors
    propagate. Caller finishes/releases plots and owns hardware/device cleanup.
    """
    readout = modules.readout.with_updates(
        tones=[modules.readout.tones[index] for index in scan.indices]
    )
    viewers = [
        context.plots.liveplot_1d(
            f"measurement_{index}", coordinate_label, "Magnitude (a.u.)"
        )
        for index in scan.indices
    ]

    def update(data: NDArray[np.complex128]) -> None:
        for row, viewer in enumerate(viewers):
            viewer.update(scan.coordinates[row], np.abs(data[:, row]))

    # Point-major storage lets Schedule write one whole tone vector per step.
    signals = SignalBuffer(scan.coordinates.T.shape, on_update=update)
    with Schedule(cfg, signals, stop=context.cancel_signal) as sched:
        for pulses, step in sched.scan("point", scan.point_pulses):
            builder = (
                step.prog_builder(context.soc, context.soccfg)
                .add_reset("reset", modules.reset)
                .add(
                    Join(
                        *[
                            Pulse(f"drive_{index}", pulse)
                            for index, pulse in zip(scan.indices, pulses, strict=True)
                        ]
                    ),
                    MuxReadout("readout", readout),
                )
            )
            prog = builder.build()
            _ = builder.run_program(prog, raw2signal_fn=partial(_averaged_signal, prog))
    signals.trigger_update(flush=True)
    return MuxSweepResult(
        np.asarray(scan.indices, dtype=np.int64),
        scan.coordinates,
        signals.array.T.copy(),
    )
