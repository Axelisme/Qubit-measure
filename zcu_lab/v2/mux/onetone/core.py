"""Fixed-tone mux spectroscopy with a common nominal host-side offset."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import field_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg, ResetCfg, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout, MuxReadoutCfg


@dataclass(frozen=True)
class MuxOnetoneResult:
    """Averaged spectra in the original cfg tone order, including partial runs.

    dfs is float64 (Ndf,) of nominal common offsets in MHz, not actual DDS
    frequencies. tone_indices is int64 (N,) containing 0..N-1; index i refers
    to the saved cfg.modules.readout.tones[i], not its logical ADC channel.
    signals is complex128 (Ndf, N) of averaged I+jQ in a.u., retaining the
    program's ADC-length normalization. Unwritten entries remain NaN; acquired
    rows survive cancellation or failure. Data alone does not indicate success:
    the run's caller must check context.cancel_signal.raise_if_error().
    """

    dfs: NDArray[np.float64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class MuxOnetoneModuleCfg(ConfigBase):
    """Modules for one fixed-tone program.

    reset is an optional native ResetCfg, default None (no reset). readout is
    a native MuxReadoutCfg with 1..8 ordered base tones: absolute freq in MHz,
    normalized gain in [-1, 1], phase in degrees and explicit logical ro_ch.
    Its generator, mixer and shared playback/ADC timing stay fixed during the
    offset scan. MuxReadoutCfg owns validation of all readout fields.
    """

    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[MuxReadoutCfg, FieldMeta(label="Readout")]


class MuxOnetoneSweepCfg(ConfigBase):
    """Common nominal frequency-offset scan, without hardware-grid rounding.

    df is SweepCfg(start, stop, expts, step) in MHz, shared by every tone.
    start, stop and step must be finite. SweepCfg requires positive expts and
    stop = start + step * (expts - 1); one point requires start = stop, step = 0.
    Invalid values raise Pydantic validation errors at construction/assignment.
    """

    df: Annotated[SweepCfg, FieldMeta(label="Frequency offset", unit="MHz")]

    @field_validator("df")
    @classmethod
    def _validate_finite_df(cls, value: SweepCfg) -> SweepCfg:
        if not all(math.isfinite(v) for v in (value.start, value.stop, value.step)):
            raise ValueError("df start, stop and step must be finite MHz values")
        return value


class MuxOnetoneCfg(ProgramV2Cfg, ExpCfgModel):
    """Inputs for a common-offset scan; hardware handles belong to RunContext.

    modules supplies the base mux readout and optional reset. sweep.df supplies
    nominal offsets in MHz; each point uses base_freq[i] + df with no cumulative
    update. Nested cfgs validate numeric settings; program/QICK validate hardware
    reachability and quantize each newly built program.

    Inherited reps (default 1) is repetitions per round; rounds (default 1) is
    software acquisition rounds per offset. initial_delay and relax_delay are
    program delays in us (both default 1). No external devices are required.
    Keep this original cfg with the Result when constructing a RunRecord so
    persistence records base frequencies and the sweep, not the final point.
    """

    modules: MuxOnetoneModuleCfg
    sweep: MuxOnetoneSweepCfg


def _mux_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    # Builder deep-copies modules; only the program-owned instance is initialized.
    for module in prog.modules:
        if module.name == "readout" and isinstance(module, MuxReadout):
            iq = module.collect_iq(prog, acquired, read_index=0)
            return iq[:, 0] + 1j * iq[:, 1]
    raise ValueError("Mux program is missing its named readout MuxReadout")


class MuxOnetoneExp(PersistableExperiment[MuxOnetoneResult, MuxOnetoneCfg]):
    """Stateless core-only spectroscopy, with one complex Signal and numeric axes.

    AXES_SPEC maps inner-first Tone (unitless int64) then Frequency offset
    (MHz in memory, Hz on disk), preserving signals.shape == (Ndf, N).
    Inherited save_run/load_run use native data; save exports one Labber complex
    log. Labber reads use zcu_tools.datafile.load_labber_data, not core.load.
    This class neither connects instruments nor registers frontend recipes.
    """

    AXES_SPEC = AxesSpec(
        axes=(
            Axis("tone_indices", "Tone", "", scale=IDENTITY, dtype=np.int64),
            Axis("dfs", "Frequency offset", "Hz", scale=MHZ_TO_HZ),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=MuxOnetoneResult,
        cfg_type=MuxOnetoneCfg,
        cfg_schema_version="2.0",
        tag="mux/onetone",
    )

    def run(self, config: MuxOnetoneCfg, *, context: RunContext) -> MuxOnetoneResult:
        """Scan fixed programs using this run's explicit hardware, plots and stop.

        config is a validated base-tone cfg and is never mutated. context must
        provide compatible soc/soccfg, active Plots and a fresh cancel_signal;
        devices may be empty. Publish N magnitude-vs-nominal-offset lines as
        measurement, and return all coordinates plus the complex signal buffer.
        No result is cached, saved or wrapped in a RunRecord here.

        Schedule propagates cancellation to acquire and retains completed data,
        leaving unwritten entries NaN. Acquire failures remain on the shared
        signal; consumers must call context.cancel_signal.raise_if_error() to
        distinguish them from successful or stopped partial returns. Cfg, build
        and other errors outside the acquire boundary propagate unchanged.
        Hardware spectra, clipping, crosstalk and synchronization are not checked.
        """
        cfg = deepcopy(config)
        base_readout = cfg.modules.readout
        dfs = sweep2array(cfg.sweep.df, "none")
        tone_indices = np.arange(len(base_readout.tones), dtype=np.int64)
        viewer = context.plots.liveplot_1d(
            "measurement",
            "Nominal frequency offset (MHz)",
            "Magnitude (a.u.)",
            num_lines=len(tone_indices),
        )
        signals_buffer = SignalBuffer(
            (len(dfs), len(tone_indices)),
            on_update=lambda data: viewer.update(dfs, np.abs(data).T),
        )
        with Schedule(cfg, signals_buffer, stop=context.cancel_signal) as sched:
            for df, step in sched.scan("df", dfs.tolist()):
                step.cfg.modules.readout = base_readout.with_updates(
                    tones=[
                        tone.with_updates(freq=tone.freq + df)
                        for tone in base_readout.tones
                    ]
                )
                builder = (
                    step.prog_builder(context.soc, context.soccfg)
                    .add_reset("reset", step.cfg.modules.reset)
                    .add(MuxReadout("readout", step.cfg.modules.readout))
                )
                prog = builder.build()
                _ = builder.run_program(prog, raw2signal_fn=partial(_mux_signal, prog))

        return MuxOnetoneResult(
            dfs=dfs,
            tone_indices=tone_indices,
            signals=signals_buffer.array,
        )
