"""Measure mux Hahn Echo coherence with simultaneous pulses and shared waits.

Technical reference for lab authors and framework developers.

`MuxT2EchoExp.run(config, *, context)` measures Hahn Echo on selected qubits in
one program with independent drives and a common mux readout. It does not create
a Form, recipe, catalog entry or writeback proposal.

## Configuration and timing

`participants` is a nonempty, unique list of names in the complete `tones` list.
Q3/Q1 selects original indices `[2, 0]`, retaining that execution/data order.
Every element requires valid drives on independent generators, even when not
participating. Names, count and order match the fixed capture.

Import `MuxT2EchoCfg`, `MuxT2EchoExp` and `MuxEchoToneCfg` from
`zcu_lab.v2.mux.t2echo`. Supply common `MuxDriveModuleCfg` with optional reset
and full readout, a full ordered `tones` list, and `delay: SweepCfg` in us.

Each tone contains fixed `name`, required `drive` for x90, required `pi_pulse`
for x180, and `detune` in MHz. Both pulses are validated for nonparticipants too.
A tone's two pulses use the same generator, but may have different waveform,
carrier, gain and phase. Active tones use independent generators, distinct from
readout. All pulse parameters must be fixed and finite. Delay bounds must be
finite and nonnegative. There is only one delay range and one set of shared
reps/rounds/initial_delay/relax_delay. Time fields use us.
The caller resolves component identities and tone/qubit bindings.

The sequence is:

1. Common reset.
2. Join of first x90 pulses.
3. Common half of requested total delay.
4. Join of x180 pulses.
5. Common half of requested total delay.
6. Join of final x90 pulses.
7. Active mux readout after the longest final stage.

All branches start at the same stage origin. Native pre/post delays remain
unchanged. Each half-delay starts after the longest preceding stage, not after
each branch's own pulse. With zero pre/post delays, a short pulse waits an extra
fixed `max_len - len_i` at that stage. This can change amplitude and constant
phase, including the balance of the two echo intervals. Analysis still fits the
common requested total delay. It does not fit an inferred per-tone actual delay.
Native clock differences of a few ticks remain. The core does not end-align, pad,
double-compile or alter the simulator.

The final x90 phase uses each source phase and its own nominal detune times the
shared total-delay parameter. Existing owners quantize pulse, phase and time.
`actual_detune(..., time_scaler=0.5)` preserves the single-qubit two-half-delay
convention. Its estimate keeps native phase wrapping, including for signed
detunes. It is not automatically a signed unwrapped correction.

## Run and analysis

```python
from zcu_tools.experiment.records import RunRecord
from zcu_lab.v2.mux.t2echo import MuxT2EchoExp

core = MuxT2EchoExp()
result = core.run(cfg, context=context)
context.cancel_signal.raise_if_error()
source = RunRecord(cfg=cfg, result=result)
outcomes = core.analyze(
    source, core.Options(fit_method="fringe", fit_phase=True), plots=analysis_plots
)
```

RunContext supplies compatible soc/soccfg, a fresh StopSignal and caller-owned
Plots. The caller finishes/releases plots and cleans up devices. Run does not
save files or retain a previous source.

`MuxCoherenceResult` contains original active `tone_indices (K,)`, requested total
`times (P,)` in us, complex average `signals (K,P)` in ADC-length-normalized a.u.,
`actual_detunes (K,)` and x90 `source_frequencies (K,)` in MHz. Full pulse cfg and
inactive tones remain in the source record. A nonzero detune with no positive
rounded delay can produce NaN actual detune. Save/load and analysis preserve that
owner output and its diagnostics, without guessing zero or nominal detune.

The hardware sweeps delay and averages reps/rounds. Cancellation preserves the
mean of completed rounds over the whole sweep; no completed round leaves IQ NaN.
Call `raise_if_error()` before success, then inspect `is_set()` for cancellation.
Acquisition failures retain their cause even with partial IQ. Validation, build
and native conversion errors propagate directly.

Analysis accepts explicit RunRecord, including cfg=None. Options reuse
`T2EchoAnalyzeOptions`: fit_method defaults to decay. Fringe requires nonzero
source detune: use cfg when present, otherwise finite saved actual detune.
Zero detune forces decay even when fringe was requested. Unknown detune fails
fringe analysis with a source diagnostic; explicit decay remains available.
The original first point is omitted before per-tone missing-IQ filtering.
Fixed fringe phase is zero; fit_phase=True frees it. Decay ignores fit_phase.
Fringe needs six remaining points and decay four. The existing fitters and
FitQuality own the numerical model and diagnostics.

Each original index returns `MuxCoherenceFit` with T2E/error in us, fringe
rate/error in MHz, selected model and quality, or `MuxFitFailure` with its reason.
A failed tone does not remove another tone's analysis. Bad source contracts
raise instead of becoming numerical fit failures. Figure names
`measurement_i` and `fit_i` use original indices, not possibly duplicate labels.
Small or poorly calibrated scans may fit numerically without supporting a useful
coherence estimate.

## Native source persistence

`save_run(source, path, metadata=...)` saves the full cfg, shared requested delay,
original indices, IQ and per-tone actual detune/source frequency metadata.
Disk units are seconds/Hz; memory units are us/MHz.
`save_run` writes only native data and returns None. `save(source, path)` is
the separate Labber export. The grouped axes spec declares the Labber
`independent` layout. Both formats preserve IQ on the delay/tone grid and detune
and source frequency on the tone-only grid. The Labber writer follows this
declaration without an automatic grid fallback, broadcasting per-tone values
or splitting files.
For multi-output saving and per-output SaveReport status/errors, see
`save_artifacts` in lib/zcu_tools/experiment/README.md, section "實驗介面與資料".
Recipe completion alone does not prove that every output saved.
`load_run(path)` restores the source without reading a current cfg or Form.
The caller sets complete/partial/stopped from the run outcome.

This core makes no calibration proposal. A later caller can use a usable Echo
fit for T2E, independently of unavailable detune. NaN actual detune must never be
replaced by a guessed frequency correction.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import model_validator
from zcu_tools.cfg_model import FieldMeta
from zcu_tools.experiment import PersistableExperiment
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
    Pulse,
    PulseCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.program.v2.modules import Join, MuxReadout

from zcu_lab.v2._support.mux import MuxFitFailure
from zcu_lab.v2._support.mux_coherence import (
    MuxCoherenceResult,
    MuxCoherenceToneCfg,
    validate_coherence_result,
    validate_coherence_scan,
)
from zcu_lab.v2._support.mux_coherence_analysis import (
    MuxCoherenceFit,
    fit_coherence_trace,
    prepare_coherence_trace,
    project_coherence_trace,
)
from zcu_lab.v2._support.mux_coherence_persistence import coherence_spec
from zcu_lab.v2._support.mux_sweep_cfg import MuxDriveModuleCfg, participant_indices
from zcu_lab.v2.twotone.time_domain.detune import actual_detune
from zcu_lab.v2.twotone.time_domain.t2echo.core import T2EchoAnalyzeOptions


class MuxEchoToneCfg(MuxCoherenceToneCfg):
    """One named qubit's x90 drive, MHz detune and x180 pi_pulse.

    Both pulses use the same generator, independent of the other qubits.
    Their frequency, gain, phase and waveform may differ. Nonparticipating
    tones still need valid pulses; name identifies the qubit in saved results.
    """

    pi_pulse: PulseCfg


class MuxT2EchoCfg(ProgramV2Cfg, ExpCfgModel):
    """Mux Hahn Echo pulses, participating qubits and a shared total-delay scan.

    modules holds optional reset and the complete mux readout. tones contains
    one uniquely named MuxEchoToneCfg per readout position. participants is a
    nonempty list of distinct names in acquisition order: Q3/Q1 gives result
    rows for original indices 2/0. All drives must use independent generators
    distinct from readout, including tones that do not participate.
    delay is a finite nonnegative total wait in us, split into two native half
    waits. Each tone's detune ramps its final x90 phase. reps/rounds are positive;
    initial_delay/relax_delay are finite nonnegative us shared by all tones.
    """

    modules: MuxDriveModuleCfg
    tones: Annotated[list[MuxEchoToneCfg], FieldMeta(element_label="name")]
    participants: list[str]
    delay: SweepCfg

    @model_validator(mode="after")
    def validate_scan(self) -> MuxT2EchoCfg:
        """Reject missing active pulses, swept values, shared channels or bad delays."""
        validate_coherence_scan(
            self, self.modules, self.tones, self.delay, self.participants
        )
        validate_coherence_scan(
            self,
            self.modules,
            [
                MuxCoherenceToneCfg(name=t.name, drive=t.pi_pulse, detune=t.detune)
                for t in self.tones
            ],
            self.delay,
            self.participants,
        )
        for index, tone in enumerate(self.tones):
            if tone.drive.ch != tone.pi_pulse.ch:
                raise ValueError(f"tones[{index}].pi_pulse.ch must equal drive.ch")
        return self


def _averaged_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_iq(prog, acquired, read_index=0)
            # Readout returns (delay, tone, I/Q); results keep tone rows.
            return (iq[..., 0] + 1j * iq[..., 1]).T
    raise ValueError("Coherence program is missing its named MuxReadout")


class MuxT2EchoExp(PersistableExperiment[MuxCoherenceResult, MuxT2EchoCfg]):
    """Measure and fit each participating qubit's Hahn Echo decay or fringes."""

    AXES_SPEC = coherence_spec(MuxT2EchoCfg, tag="mux/time_domain/t2echo")
    Options: ClassVar[type[T2EchoAnalyzeOptions]] = T2EchoAnalyzeOptions

    def run(self, config: MuxT2EchoCfg, *, context: RunContext) -> MuxCoherenceResult:
        """Measure x90 → half wait → x180 → half wait → x90 → mux readout.

        config supplies full tones, ordered participants and a total delay sweep
        in us. Each pulse stage starts together; the next wait or readout starts
        after its longest pulse. The final x90 phase ramps with each tone's MHz
        detune. Return requested total delays (P,), averaged complex IQ (K,P),
        original tone indices, actual detunes and source frequencies (K,).
        context supplies hardware and live plots. Cancellation keeps acquired
        averages and NaN gaps; call cancel_signal.raise_if_error before success.
        """
        # 1. Keep participant order while selecting from the complete tone list.
        cfg = MuxT2EchoCfg.model_validate(config.model_dump())
        indices = participant_indices(
            [tone.name for tone in cfg.tones], cfg.participants
        )
        tones = [cfg.tones[index] for index in indices]
        times = sweep2array(cfg.delay, "none")
        # Both waits use the native half-delay grid, not a rounded total delay.
        _ = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg, "scaler": 0.5})
        detunes = np.asarray(
            [
                actual_detune(
                    cfg.delay,
                    detune=tone.detune,
                    gen_ch=tone.drive.ch,
                    soccfg=context.soccfg,
                    time_scaler=0.5,
                )
                for tone in tones
            ],
            dtype=np.float64,
        )
        frequencies = np.asarray(
            [float(tone.drive.freq) for tone in tones], dtype=np.float64
        )
        readout = cfg.modules.readout.with_updates(
            tones=[cfg.modules.readout.tones[index] for index in indices]
        )

        # 2. Show each selected tone on the common requested total-delay axis.
        viewers = [
            context.plots.liveplot_1d(
                f"measurement_{index}",
                "Requested delay (us)",
                "Magnitude (a.u.)",
                title=f"{index}: {cfg.tones[index].name}",
            )
            for index in indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for row, viewer in enumerate(viewers):
                viewer.update(times, np.abs(data[row]))

        signals = SignalBuffer((len(indices), times.size), on_update=update)
        with Schedule(cfg, signals, stop=context.cancel_signal) as sched:
            # 3. Join each pulse stage so both shared waits start after all drives.
            length = sweep2param("length", cfg.delay)
            builder = (
                sched.prog_builder(context.soc, context.soccfg)
                .add_reset("reset", cfg.modules.reset)
                .add(
                    Join(
                        *[
                            Pulse(f"pi2_first_{index}", tone.drive)
                            for index, tone in zip(indices, tones, strict=True)
                        ]
                    ),
                    Delay("delay_first", 0.5 * length),
                    Join(
                        *[
                            Pulse(f"pi_{index}", tone.pi_pulse)
                            for index, tone in zip(indices, tones, strict=True)
                        ]
                    ),
                    Delay("delay_second", 0.5 * length),
                    Join(
                        *[
                            Pulse(
                                f"pi2_final_{index}",
                                tone.drive.with_updates(
                                    phase=tone.drive.phase + 360 * tone.detune * length
                                ),
                            )
                            for index, tone in zip(indices, tones, strict=True)
                        ]
                    ),
                    MuxReadout("readout", readout),
                )
                .declare_sweep("length", cfg.delay)
            )

            # 4. Average completed rounds into tone-major IQ, preserving NaN gaps.
            prog = builder.build()
            _ = builder.run_program(prog, raw2signal_fn=partial(_averaged_signal, prog))
        signals.trigger_update(flush=True)
        return MuxCoherenceResult(
            np.asarray(indices, dtype=np.int64),
            times,
            signals.array.copy(),
            detunes,
            frequencies,
        )

    def analyze(
        self,
        source: RunRecord[MuxT2EchoCfg, MuxCoherenceResult],
        options: T2EchoAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxCoherenceFit | MuxFitFailure, ...]:
        """Fit each source tone's Echo decay or fringes against requested delay.

        source is a Run/Load record, with optional cfg. options.fit_method selects
        decay or fringe; zero source detune forces decay. Gating uses cfg detune
        when present, otherwise finite saved actual_detunes. Unknown detune fails
        a requested fringe fit with a missing-source reason, not a guessed model.
        Explicit decay needs no detune. fit_phase frees fringe phase (otherwise zero).
        Omit the first point and missing IQ. Return per-tone T2E in us, fringe
        rate in MHz, uncertainties and fit quality, or a fit failure reason.
        Publish fit_i plots; invalid sources or fit methods raise.
        """
        # 1. Omit the first requested point before removing missing IQ.
        if options.fit_method not in ("fringe", "decay"):
            raise ValueError(f"Unknown fit_method: {options.fit_method}")
        validate_coherence_result(source.result)
        outcomes: list[MuxCoherenceFit | MuxFitFailure] = []
        for row, raw_index in enumerate(source.result.tone_indices):
            trace = prepare_coherence_trace(
                int(raw_index), source.result.times, source.result.signals[row], skip=1
            )
            fig, ax = plots.subplots(f"fit_{trace.index}", figsize=(8, 5))
            ax.set(xlabel="Requested delay (us)", ylabel="Signal real (a.u.)")
            fig.suptitle(f"echo tone {trace.index}")
            try:
                # Source detune, not a fitted oscillation, determines fringe eligibility.
                fit_fringe = options.fit_method == "fringe"
                if fit_fringe:
                    detune = (
                        source.cfg.tones[trace.index].detune
                        if source.cfg is not None
                        else float(source.result.actual_detunes[row])
                    )
                    if not np.isfinite(detune):
                        raise ValueError(
                            "Echo fringe fit requires source detune: cfg or saved finite actual_detune"
                        )
                    fit_fringe = detune != 0.0
                real = project_coherence_trace(
                    trace, minimum_points=6 if fit_fringe else 4
                )

                # 2. Echo fringes start at zero phase unless phase is fitted freely.
                phase = None if options.fit_phase else 0.0
                outcome, fitted = fit_coherence_trace(
                    trace,
                    real,
                    fit_fringe=fit_fringe,
                    phase=phase,
                    expected_frequency_mhz=None,
                )
            except (
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                ax.plot(
                    trace.times, np.abs(trace.signal), "o", label="acquired magnitude"
                )
                ax.text(
                    0.02, 0.98, str(exc), transform=ax.transAxes, va="top", wrap=True
                )
                outcomes.append(MuxFitFailure(trace.index, str(exc)))
            else:
                ax.plot(trace.times, real, "o", label="measured")
                ax.plot(trace.times, fitted, label="fit")
                ax.set_title(
                    f"T2 = {outcome.t2:.4g} ± {outcome.t2_err:.2g} us; "
                    f"fringe = {outcome.detune:.4g} MHz"
                )
                outcomes.append(outcome)

            # 3. Keep each tone's fit or failure visible without stopping other fits.
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
        return tuple(outcomes)
