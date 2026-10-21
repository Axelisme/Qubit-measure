"""Simultaneous T1 acquisition on one requested delay axis.

Technical reference for lab authors and framework developers.

## Sequence and coordinates

The complete `tones` list retains capture-generated `name` values and original
readout positions. `participants` is a nonempty, unique list of existing names.
Q3/Q1 selects original indices `[2, 0]`, retaining that execution/data order.
Every element requires valid drives, including nonparticipants.

`MuxT1Exp` excites active qubits with fixed pi pulses on independent drive generators.
It uses the existing `Join` to start all pulses together. Each shorter branch waits
for the longest branch to finish. One common `Delay` follows the Join, then one
active mux readout. Reset, reps, rounds, initial delay and relax delay are shared.
Pulse pre/post delays remain part of their native branches; no end-alignment
padding or second compilation is added.

`MuxT1Cfg.delays` is the only delay axis. Supply a `SweepCfg` or an explicit list
of finite, nonnegative, strictly increasing requested times in microseconds.
All qubits use the same requested axis. The program owns time quantization, so
readout timing can differ from the requested delay by a few native ticks.
The result and fit retain requested times, not a reconstructed actual-delay axis.
A list can describe a nonuniform scan; the core does not choose a sampling model.

A short pi pulse adds a fixed `max_len - len_i` wait before the common delay.
For an exponential decay this changes the amplitude, not T1. Pulse delays can
add their own fixed offsets. Calibrate the pulses first and choose a delay window
long enough to constrain the decay and baseline. This core does not implement
T1 with a probe tone or singleshot T1.

## Inputs and ownership

Import the leaf without loading a GUI:

```python
from zcu_lab.v2._support.mux_sweep_cfg import MuxDriveModuleCfg
from zcu_lab.v2.mux.t1 import MuxT1Cfg, MuxT1Exp, MuxT1ToneCfg
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.plotting.plots import NonPresentingHost, Plots

# env is a caller-owned compatible hardware or simulation environment.
# full_readout is MuxReadoutCfg; pi_0/pi_1/pi_2 are fixed PulseCfg values.
# Their generators must be independent of each other and readout.gen_ch.
cfg = MuxT1Cfg(
    modules=MuxDriveModuleCfg(readout=full_readout),
    tones=[
        MuxT1ToneCfg(name="Q1", drive=pi_0),
        MuxT1ToneCfg(name="Q2", drive=pi_1),
        MuxT1ToneCfg(name="Q3", drive=pi_2),
    ],
    participants=["Q3", "Q1"],
    delays=[0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0],
    reps=128,
    rounds=2,
    initial_delay=1.0,
    relax_delay=40.0,
)
core = MuxT1Exp()
plots = Plots(NonPresentingHost())
stop = StopSignal()
try:
    result = core.run(
        cfg,
        context=RunContext(env.soc, env.soccfg, plots, {}, stop),
    )
    stop.raise_if_error()
    source = RunRecord(cfg=cfg, result=result)
    fits = core.analyze(source, core.Options(skip=0, dual_exp=False), plots=plots)
    # Preserve completion separately: stop.is_set() means cancelled, not complete.
    # metadata is caller-captured RunMetadata for this source and its completion.
    core.save_run(source, destination, metadata=metadata)
    loaded, snapshot = core.load_run(destination)
finally:
    plots.finish(present=False)
    plots.release()
# The environment owner closes its devices after all operations finish.
```

The full ordered tone list must match the full readout count. Every tone requires
a valid drive, including nonparticipants. The nonempty, unique participants name
list selects original indices in its own order;
ADC channel order and duplicate labels do not change identity. Cfg validation
rejects missing drives, shared generators, invalid counts and invalid delay axes.
Native pulse and readout owners reject unsupported hardware settings.

The caller owns compatible soc/soccfg, devices, plots and a fresh stop signal.
This core does not initialize external devices, save automatically or close a
caller's environment. Use `ftf(settings, workers=1)` for a physical offline smoke,
with typed YAML `readout.drive.kind: mux`, an ADC receiver list and independently
bound drive lines. The FTF backend is `ftf-segment-v7`; earlier replay backends
are rejected, not converted. Small truncations demonstrate execution only and
do not establish physical convergence or calibration accuracy.

## Results, cancellation and analysis

`MuxT1Result.times` is float64 `(P,)` requested us. `tone_indices` is int64 `(K,)`
of original active indices. `signals` is complex128 `(K,P)`, ADC-length-normalized
average I+jQ. The program averages reps and rounds. Cancellation keeps completed
points and the current point's completed-round average. Unwritten points remain
NaN, including a stop before any data. Acquisition failures retain their cause
on `StopSignal`; always call `raise_if_error()` before reporting success. A
partial buffer does not imply success. Admission and construction errors can
propagate directly.

Live plots use `measurement_i` and fit plots use `fit_i`, where `i` is the original
index, not the label. Analysis reads only the explicit source, permits cfg=None,
and removes `skip` leading points before finite-IQ filtering. It reuses the
single-qubit T1 rotation, single/dual decay fitters and fit-quality calculation.
`MuxT1Fit(index, fit)` contains the existing `T1Analysis`, including microsecond
values/errors, optional second decay and quality under `fit`. A numerical failure
returns `MuxFitFailure(index, reason)` without hiding another tone's outcome.
Malformed source shapes or options raise instead of becoming fit failures.
Inspect quality and uncertainty before using a fitted value as calibration.

Native save/load preserve full cfg, original indices, complex IQ and the one
requested axis. Time is seconds on disk and microseconds in memory. Always pass
the captured `RunRecord`; changing a current cfg does not change the saved source.
Canonical saving requires cfg. `save_run`/`load_run` use native HDF5; inherited
`save` is a Labber export, not the inverse of native `load`.

The numerical core does not infer component identity from readout labels.
The adapter below owns component proposals using the captured binding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import US_TO_S, AxesSpec, Axis, PersistableExperiment, ZSpec
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    ProgramV2Cfg,
    PulseCfg,
    SweepCfg,
)
from zcu_tools.program.v2.modules import (
    ConstWaveformCfg,
    FlatTopWaveformCfg,
    Join,
    MuxReadout,
    Pulse,
)

from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2._support.mux_sweep_cfg import MuxDriveModuleCfg, participant_indices
from zcu_lab.v2.twotone.time_domain.t1.core import T1AnalyzeOptions

from .analysis import MuxT1Fit, analyze_t1


class MuxT1ToneCfg(ConfigBase):
    """One named original tone's required fixed pi pulse.

    name is the nonempty capture-generated qubit name, unique in the complete
    list. drive is required and validated even for nonparticipants: fixed finite
    MHz frequency, normalized gain, degree phase and nonnegative us delays.
    Native waveform schemas own shape/clock conversion; length cannot be swept.
    """

    name: Annotated[str, FieldMeta(readonly=True)]
    drive: PulseCfg

    @model_validator(mode="after")
    def validate_drive(self) -> Self:
        """Reject swept or nonfinite pulses, including nonparticipants."""
        for name in ("freq", "gain", "phase", "pre_delay", "post_delay"):
            value = getattr(self.drive, name)
            if isinstance(value, QickParam) or not math.isfinite(value):
                raise ValueError(f"drive.{name} must be fixed and finite")
        if self.drive.pre_delay < 0 or self.drive.post_delay < 0:
            raise ValueError("drive delays must be nonnegative")
        if isinstance(
            self.drive.waveform, (ConstWaveformCfg, FlatTopWaveformCfg)
        ) and isinstance(self.drive.waveform.length, QickParam):
            raise ValueError("drive.waveform.length must be fixed")
        return self


class MuxT1Cfg(ProgramV2Cfg, ExpCfgModel):
    """Complete named tones, ordered participants and shared acquisition controls.

    modules holds optional reset and complete mux readout. tones contains one
    MuxT1ToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of those names; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Every drive is required/validated, including nonparticipants, on independent
    generators distinct from readout. delays is one common SweepCfg or strictly increasing explicit list of finite
    nonnegative requested us. Fits retain requested times, not compiled delays.
    Inherited reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    modules: MuxDriveModuleCfg
    tones: Annotated[list[MuxT1ToneCfg], FieldMeta(element_label="name")]
    participants: list[str]
    delays: SweepCfg | list[float]

    @model_validator(mode="after")
    def validate_scan(self) -> Self:
        """Reject invalid counts, channels and requested delay coordinates."""
        if self.reps <= 0 or self.rounds <= 0:
            raise ValueError("reps and rounds must be positive")
        if any(
            not math.isfinite(v) or v < 0
            for v in (self.initial_delay, self.relax_delay)
        ):
            raise ValueError(
                "initial_delay and relax_delay must be finite nonnegative us"
            )
        if len(self.tones) != len(self.modules.readout.tones):
            raise ValueError("tones must match the full readout tone count")
        participant_indices([tone.name for tone in self.tones], self.participants)
        channels: set[int] = set()
        for index, tone in enumerate(self.tones):
            drive = tone.drive
            if drive.ch == self.modules.readout.gen_ch or drive.ch in channels:
                raise ValueError(
                    f"tones[{index}].drive.ch requires an independent generator"
                )
            channels.add(drive.ch)
        times = sweep2array(self.delays, "none", allow_array=True)
        if (
            times.ndim != 1
            or not times.size
            or not np.all(np.isfinite(times))
            or np.any(times < 0)
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("delays must be nonempty finite nonnegative increasing us")
        return self


@dataclass(frozen=True)
class MuxT1Result:
    """Averaged complex IQ on a common requested delay axis.

    times is float64 (P,) requested us, not compiled actual delays. tone_indices
    is int64 (K,) of original full-tone indices in participants order.
    signals is complex128 (K,P), I+jQ in ADC-length-normalized a.u. Unwritten
    entries remain NaN; completed-round means survive cancellation. The caller
    must check the RunContext stop signal before declaring success.
    """

    times: NDArray[np.float64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


def _averaged_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    # Builder copies modules; collection must use the initialized program owner.
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_iq(prog, acquired, read_index=0)
            return iq[:, 0] + 1j * iq[:, 1]
    raise ValueError("T1 program is missing its named MuxReadout")


class MuxT1Exp(PersistableExperiment[MuxT1Result, MuxT1Cfg]):
    """Measure selected qubits' energy relaxation on one shared delay axis.

    Each point applies simultaneous pi pulses, waits for the longest branch,
    then waits the requested delay before a joint mux readout. Saved results
    retain requested times, original tone indices, complex IQ and the full cfg.
    """

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("times", "Requested delay", "s", US_TO_S),
            Axis("tone_indices", "Tone", "", dtype=np.int64),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=MuxT1Result,
        cfg_type=MuxT1Cfg,
        cfg_schema_version="1.0",
        tag="mux/t1",
    )

    def run(self, config: MuxT1Cfg, *, context: RunContext) -> MuxT1Result:
        """Acquire pi pulse -> shared wait -> mux readout for each delay in us.

        config supplies the full tone list, ordered participants, pi pulses,
        reset/readout and averaging controls; it is validated without mutation.
        Join waits for the longest pi branch, including its pulse delays, before
        the shared wait. Return requested times (P,), original tone indices (K,)
        and averaged complex IQ (K,P), with one live figure per tone.
        Cancelled scans retain acquired averages and leave unwritten IQ as NaN.
        context supplies offline or hardware handles, plots and a stop signal;
        check cancel_signal.raise_if_error() before reporting success.
        """
        # 1. Keep participant order while selecting readout tones from the full cfg.
        cfg = MuxT1Cfg.model_validate(config.model_dump())
        indices = participant_indices(
            [tone.name for tone in cfg.tones], cfg.participants
        )
        times = sweep2array(cfg.delays, "none", allow_array=True)
        readout = cfg.modules.readout.with_updates(
            tones=[cfg.modules.readout.tones[i] for i in indices]
        )

        # 2. Show each decay in its own figure on the requested delay axis.
        viewers = [
            context.plots.liveplot_1d(
                f"measurement_{i}",
                "Requested delay (us)",
                "Magnitude (a.u.)",
                title=f"{i}: {cfg.tones[i].name}",
            )
            for i in indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for row, viewer in enumerate(viewers):
                viewer.update(times, np.abs(data[:, row]))

        # 3. Compile each delay once; all pi pulses start together after reset.
        signals = SignalBuffer((len(times), len(indices)), on_update=update)
        with Schedule(cfg, signals, stop=context.cancel_signal) as sched:
            for delay, step in sched.scan("delay", times.tolist()):
                pulses = []
                for index in indices:
                    drive = cfg.tones[index].drive
                    pulses.append(Pulse(f"drive_{index}", drive))
                builder = (
                    step.prog_builder(context.soc, context.soccfg)
                    .add_reset("reset", cfg.modules.reset)
                    .add(
                        Join(*pulses),
                        Delay("t1_delay", float(delay)),
                        MuxReadout("readout", readout),
                    )
                )
                prog = builder.build()
                # Average completed rounds into this delay's participant rows.
                _ = builder.run_program(
                    prog, raw2signal_fn=partial(_averaged_signal, prog)
                )

        # 4. Flush live plots and return one IQ row per original selected tone.
        signals.trigger_update(flush=True)
        return MuxT1Result(
            times, np.asarray(indices, dtype=np.int64), signals.array.T.copy()
        )

    def analyze(
        self,
        source: RunRecord[MuxT1Cfg, MuxT1Result],
        options: T1AnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxT1Fit | MuxFitFailure, ...]:
        """Fit each saved IQ row against the requested delays to estimate T1 in us.

        source is a Run/Load record; its cfg may be absent. options selects
        initial points to skip and single or dual exponential decay. Return
        per-tone fits with uncertainties and quality, or indexed fit failures;
        invalid axes/options raise. Draw each decay in plots as fit_i.
        """
        return analyze_t1(source.result, options, plots=plots)
