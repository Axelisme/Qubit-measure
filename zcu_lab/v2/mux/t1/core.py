"""Simultaneous T1 acquisition on one shared delay axis.

Technical reference for lab authors and framework developers.

## Sequence and coordinates

The complete `tones` list retains capture-generated `name` values and original
readout positions. `participants` is a nonempty, unique list of original indices.
`[2, 0]` selects Q3 then Q1, retaining that execution/data order.
Only participants require valid drives. Inactive drives may be None.

`MuxT1Exp` excites active qubits with fixed pi pulses on independent drive generators.
It uses the existing `Join` to start all pulses together. Each shorter branch waits
for the longest branch to finish. One common `Delay` follows the Join, then one
active mux readout. Reset, reps, rounds, initial delay and relax delay are shared.
Pulse pre/post delays remain part of their native branches; no end-alignment
padding or second compilation is added.

`MuxT1Cfg.delays` defines a finite, nonnegative, strictly increasing start/stop
window and point count in microseconds. `uniform=True` uses its linear SweepCfg
axis in one hardware loop. The result and fit retain requested times; readout
can differ by native timing ticks. `uniform=False` uses the single-qubit T1
sampling policy to concentrate points in the decay region of the same window.
LoadValue and DelayAuto execute one shared cycle table for all qubits. Results
and fits then retain quantized delays; duplicate/reordered quantized points
raise ValueError before acquisition.

A short pi pulse adds a fixed `max_len - len_i` wait before the common delay.
For an exponential decay this changes the amplitude, not T1. Pulse delays can
add their own fixed offsets. Calibrate the pulses first and choose a delay window
long enough to constrain the decay and baseline. This core does not implement
T1 with a probe tone or singleshot T1.

## Inputs and ownership

Import the leaf without loading a GUI:

```python
from zcu_lab.v2.support.mux_sweep_cfg import MuxDriveModuleCfg
from zcu_lab.v2.mux.t1 import MuxT1Cfg, MuxT1Exp, MuxT1ToneCfg
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment import load_run, save_run
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import SweepCfg

# env is a caller-owned compatible hardware or simulation environment.
# full_readout is MuxReadoutCfg; pi_0/pi_1/pi_2 are fixed PulseCfg values.
# Their generators must be independent of each other and readout.gen_ch.
cfg = MuxT1Cfg(
    readout=full_readout,
    tones=[
        MuxT1ToneCfg(name="Q1", drive=pi_0),
        MuxT1ToneCfg(name="Q2", drive=pi_1),
        MuxT1ToneCfg(name="Q3", drive=pi_2),
    ],
    participants=[2, 0],
    delays=SweepCfg(start=0.0, stop=8.0, expts=41, step=0.2),
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
    source = RunRecord(cfg=cfg, result=result, metadata=metadata)
    fits = core.analyze(source, core.Options(skip=0, dual_exp=False), plots=plots)
    # Preserve completion separately: stop.is_set() means cancelled, not complete.
    # metadata is caller-captured RunMetadata for this source and its completion.
    save_run(source, destination, spec=MuxT1Exp.AXES_SPEC)
    loaded, snapshot = load_run(destination, spec=MuxT1Exp.AXES_SPEC)
finally:
    plots.finish(present=False)
    plots.release()
# The environment owner closes its devices after all operations finish.
```

The full ordered tone list must match the full readout count. Only participants
require valid drives. The nonempty, unique participants list contains original
indices in its own order;
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

`MuxT1Result.times` is float64 `(P,)` us, requested for uniform sampling and
quantized for nonuniform sampling. `tone_indices` is int64 `(K,)`
of original active indices. `signals` is complex128 `(K,P)`, ADC-length-normalized
average I+jQ. The program averages reps and rounds. Cancellation keeps
completed-round averages across the whole sweep. Before the first completed
round, signals remain NaN. Acquisition failures retain their cause on `StopSignal`;
always call `raise_if_error()` before reporting success. A
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
recorded delay axis. Time is seconds on disk and microseconds in memory. Always pass
the captured `RunRecord`; changing a current cfg does not change the saved source.
Canonical saving requires cfg. External `save_run`/`load_run` use native HDF5;
`save_labber(source, path, spec=MuxT1Exp.AXES_SPEC)` exports Labber separately.

The numerical core does not infer component identity from readout labels.
The adapter below owns component proposals using the captured binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Annotated, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, StrictInt, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.experiment.v2.runtime.program_builder import AcquireOptions
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.t1_sampling import materialize_nonuniform_t1_delays
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Delay,
    DelayAuto,
    LoadValue,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.program.v2.modules import (
    Join,
    MuxReadout,
    MuxReadoutCfg,
    Pulse,
)

from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.acquire import mux_snr_checker
from zcu_lab.v2.support.mux import MuxFitFailure, averaged_mux_signal
from zcu_lab.v2.support.mux_sweep_cfg import MuxDriveToneCfg, validate_mux_tones
from zcu_lab.v2.twotone.time_domain.t1.core import T1AnalyzeOptions

from .analysis import MuxT1Fit, analyze_t1


class MuxT1ToneCfg(MuxDriveToneCfg):
    """One original qubit's nullable fixed pi pulse, required only when active."""


class MuxT1Cfg(ConfigBase):
    """Complete named tones, ordered participants and shared acquisition controls.

    reset and readout hold the common modules. tones contains one
    MuxT1ToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of original tone indices; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Selected drives are required/validated on independent
    generators distinct from readout. delays is one common linear SweepCfg of
    strictly increasing, finite, nonnegative requested us. uniform defaults True
    for a linear hardware sweep. False concentrates points within that window
    using a shared cycle table. Fits retain requested times for uniform sampling,
    quantized times otherwise.
    Explicit reps/rounds are positive; initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    reps: Annotated[
        int, FieldMeta(label="Reps", description="Number of repetitions per round.")
    ] = 1
    rounds: Annotated[
        int,
        FieldMeta(label="Rounds", description="Number of software acquisition rounds."),
    ] = 1
    acquire_retry: Annotated[
        int,
        FieldMeta(
            label="Acquire retry",
            description="Retries after a transient acquisition failure.",
        ),
    ] = Field(default=0, ge=0)
    earlystop_snr: Annotated[
        float | None,
        FieldMeta(
            label="Early-stop SNR",
            description="Stop when every active tone reaches this trace SNR; None disables.",
        ),
    ] = Field(default=None, gt=0)
    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay",
            unit="us",
            description="Delay before the first repetition.",
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Relaxation delay after each repetition.",
        ),
    ] = 1.0
    reset: ResetCfg | None = None
    readout: MuxReadoutCfg
    tones: Annotated[list[MuxT1ToneCfg], FieldMeta(element_label="name")]
    participants: list[StrictInt]
    delays: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    uniform: Annotated[bool, FieldMeta(label="Uniform (linear) sweep")] = True

    @model_validator(mode="after")
    def validate_scan(self) -> Self:
        """Reject invalid counts, channels and requested delay coordinates."""
        validate_mux_tones(self, self.tones, self.participants)
        times = sweep2array(self.delays, "none")
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
    """Averaged complex IQ on a common delay axis.

    times is float64 (P,) us, requested for uniform sampling and quantized for
    nonuniform sampling. tone_indices is int64 (K,) of original full-tone indices
    in participants order.
    signals is complex128 (K,P), I+jQ in ADC-length-normalized a.u. Unwritten
    entries remain NaN; completed-round means survive cancellation. The caller
    must check the RunContext stop signal before declaring success.
    """

    times: NDArray[np.float64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class MuxT1Exp(Experiment[MuxT1Result, MuxT1Cfg]):
    """Measure selected qubits' energy relaxation on one shared delay axis.

    Each point applies simultaneous pi pulses, waits for the longest branch,
    then waits a shared delay before a joint mux readout. Saved results retain
    recorded delays, original tone indices, complex IQ and the full cfg.
    """

    Options: ClassVar[type[T1AnalyzeOptions]] = T1AnalyzeOptions
    AXES_SPEC: ClassVar[AxesSpec[MuxT1Result, MuxT1Cfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("times", "Delay", "s", 1e-6),
            AxesSpec.Axis("tone_indices", "Tone", "", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u."),
        result_type=MuxT1Result,
        cfg_type=MuxT1Cfg,
        cfg_schema_version="3.0",
        tag="mux/t1",
    )

    def run(self, config: MuxT1Cfg, *, context: RunContext) -> MuxT1Result:
        """Acquire pi pulse -> shared wait -> mux readout for each delay in us.

        config supplies the full tone list, ordered participants, pi pulses,
        reset/readout and averaging controls; it is validated without mutation.
        Join waits for the longest pi branch, including its pulse delays, before
        the shared wait. Return times (P,), original tone indices (K,) and
        averaged complex IQ (K,P), with one live figure per tone. uniform=True
        retains the requested linear times; False returns quantized nonuniform
        delays in the same window using the single-qubit T1 sampling policy.
        Both paths use one hardware loop. Nonuniform quantization collisions
        raise ValueError before acquisition. Cancellation retains
        completed-round averages across the sweep, or NaN before any round finishes.
        context supplies offline or hardware handles, plots and a stop signal;
        check context.scope.stop.raise_if_error() before reporting success.
        """
        # 1. Keep participant order while selecting readout tones from the full cfg.
        cfg = MuxT1Cfg.model_validate(config.model_dump())
        indices = list(cfg.participants)
        delay_table = (
            None
            if cfg.uniform
            else materialize_nonuniform_t1_delays(cfg.delays, soccfg=context.soccfg)
        )
        times = (
            sweep2array(cfg.delays, "none")
            if delay_table is None
            else delay_table.times_us
        )
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[i] for i in indices]
        )

        # 2. Show each decay on the shared requested or quantized delay axis.
        viewers = [
            live_line(
                context.plots,
                f"measurement_{i}",
                "Delay (us)",
                "Magnitude (a.u.)",
                title=f"{i}: {cfg.tones[i].name}",
            )
            for i in indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for row, viewer in enumerate(viewers):
                viewer.update(times, np.abs(data[:, row]))

        # 3. Compile one shared delay loop; pi pulses start together after reset.
        signals = SignalBuffer((len(times), len(indices)), on_update=update)
        with Schedule(cfg, signals, scope=context.scope) as sched:
            pulses = [
                Pulse(f"drive_{index}", cfg.tones[index].require_drive())
                for index in indices
            ]
            builder = sched.prog_builder(
                context.soc,
                context.soccfg,
                reps=sched.cfg.reps,
                initial_delay=sched.cfg.initial_delay,
                relax_delay=sched.cfg.relax_delay,
            )
            if delay_table is None:
                delay = Delay("t1_delay", sweep2param("delay", cfg.delays))
                builder.declare_sweep("delay", cfg.delays)
            else:
                # One shared table drives every participant in the same hardware loop.
                builder.add(
                    LoadValue(
                        "load_t1_delay",
                        values=list(delay_table.cycles),
                        idx_reg="delay_idx",
                        val_reg="t1_delay_cycle",
                        auto_compress=False,
                    )
                )
                delay = DelayAuto("t1_delay", t="t1_delay_cycle")
                builder.declare_sweep("delay_idx", len(delay_table.cycles))
            builder.add_reset("reset", cfg.reset).add(
                Join(*pulses),
                delay,
                MuxReadout("readout", readout),
            )
            prog = builder.build()
            # Each completed round supplies the full delay/participant IQ matrix.
            _ = builder.run_program(
                prog,
                raw2signal_fn=partial(averaged_mux_signal, prog),
                rounds=cfg.rounds,
                options=AcquireOptions(
                    retry=cfg.acquire_retry,
                    stop_condition=mux_snr_checker(
                        signals, cfg.earlystop_snr, tone_axis=1
                    ),
                ),
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
        max_model_window_factor: float = 2.0,
    ) -> tuple[MuxT1Fit | MuxFitFailure, ...]:
        """Fit each saved IQ row against its recorded delays to estimate T1 in us.

        source is a Run/Load record; its cfg may be absent. options selects
        initial points to skip and single or dual exponential decay. Return
        per-tone fits with uncertainties and quality, or indexed fit failures;
        invalid axes/options raise. Single-exponential mode also records and
        draws a dual-exponential comparison; the primary T1 is unchanged.
        max_model_window_factor (default 2) marks diagnostic lifetimes above
        this multiple of max fitted delay as discarded. See fit.comparison
        for both records and numerical secondary failures. Draw fits as fit_i.
        """
        return analyze_t1(
            source.result,
            options,
            plots=plots,
            max_model_window_factor=max_model_window_factor,
        )
