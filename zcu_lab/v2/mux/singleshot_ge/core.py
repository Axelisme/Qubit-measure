"""Record simultaneous mux IQ shots with each qubit's probe pulse off and on.

Technical reference for lab authors and framework developers.

## Acquisition

The complete `tones` list retains capture-generated `name` values and original
readout positions. `participants` is a nonempty, unique list of original indices.
`[2, 0]` selects Q3 then Q1, retaining that execution/data order.
Only participants require valid drives. Inactive drives may be None.

`MuxGEExp.run(config, *, context)` acquires probe-off and probe-on raw IQ for
all active qubits in the same two programs. It does not simulate each qubit
separately or assemble unrelated shots. This core has no Form, catalog entry,
recipe or calibration proposals.

`MuxGECfg.reset` is the common optional reset. `MuxGECfg.readout` is the
complete ordered mux readout. `tones: list[MuxGEToneCfg]` has one element per full readout tone.
Each element contains fixed `name`, required probe `drive`, and optional `init_pulse`.
All drives must be valid. All generators must be independent and
must differ from the readout generator. An initial pulse uses its tone's drive
generator. Nonparticipant settings stay validated in cfg but do not run.

Each preparation runs common reset, then an existing `Join` of the active
branches, then one mux readout. A branch contains its optional initial pulse
followed by the probe in the probe-on preparation only. Branches start together;
shorter branches finish early and wait. Readout follows the longest branch.
There is no end-alignment padding or custom timing module. Native program
modules own MHz/us/degree/gain conversion and hardware quantization. The initial
pulse runs in both preparations; it is not an analysis initial-state option.

`shots` is the number of raw shots in each preparation. As in single-qubit GE,
run copies cfg and explicitly passes `reps=shots` to the builder and `rounds=1`
to acquisition. These derived counts are not experiment inputs. Caller cfg is
not changed. The acquired result represents one round per preparation, with its
actual shot count on `shot_indices`. The underlying raw getter retains only its
last round; using one round avoids silently dropping earlier rounds.

The program-owned `MuxReadout.collect_raw_iq(prog, read_index=0)` reads the
initialized module of that same acquire. It selects the same read on every tone,
preserves all reps and divides each ADC sum by its rounded integration length.
It does not subtract offsets, rotate IQ, classify shots or average them. This
experiment declares one read. No lab code parses QICK raw buffers.

## Result and cancellation

`MuxGEResult` contains:

- `signals`: complex128 `(K, 2, shots)`, in ADC units per integration sample.
- `tone_indices`: int64 `(K,)`, the active original full-readout indices in order.
- `prepared_states`: int64 `[0, 1]`, meaning probe off/on, not pure g/e states.
- `shot_indices`: int64 `arange(shots)`.

At a given preparation and shot index, all tones belong to the same acquisition
shot. ADC channel numbers and duplicate labels do not reorder these axes. The
two preparations are separate acquires. They are not four-state joint assignment
calibration and do not identify a correlated two-qubit confusion matrix.

`context` supplies compatible hardware, a fresh `StopSignal`, devices and
caller-owned `Plots`. Run publishes `measurement_i` magnitude traces with both
preparations, where `i` is the original tone index. Caller finishes/releases the
plots and closes owned devices.

Completed preparations survive cancellation or acquisition error. A missing
preparation remains NaN across every tone and shot. An incomplete first round is
not accepted as a complete preparation. After run, call
`context.scope.stop.raise_if_error()` before reporting success. Then inspect
`is_set()` to distinguish cancellation from completion. The signal retains an
acquisition error's cause even when run returns useful partial data. Validation
and build errors propagate. The caller records completion separately; a saved
partial array does not mean the operation succeeded.

## Analysis

`analyze(source, options, *, plots)` reads only the explicit `RunRecord`.
`MuxGEExp.Options()` applies the existing GE defaults to every active tone.
Use original-index `overrides` for different initial states or fit options:

```python
from zcu_lab.v2.mux.singleshot_ge import MuxGEExp
from zcu_lab.v2.singleshot.ge.core import GEAnalyzeOptions

core = MuxGEExp()
options = core.Options(
    overrides={2: GEAnalyzeOptions(initial_state="excited", backend="pca")}
)
primary = core.analyze(source, options, plots=analysis_plots)
post = core.post_analyze(source, primary, core.PostOptions(), plots=analysis_plots)
```

`source` is the retained run/load record. `analysis_plots` is a fresh
caller-owned `Plots`. `default` and each override use `GEAnalyzeOptions`:
initial state, PCA/center backend, optional angle in degrees, log histogram,
T1 alignment and optional dimensionless length ratio. Initial state describes
the predominant state after reset/init and before probe. It only maps the
analysis rows to physical g/e; stored probe-off/on data never moves.

Each original index returns `MuxGEAnalysis(index, primary)` or
`MuxFitFailure(index, reason)`, with a named `fit_i` figure. The single-qubit GE
owner computes centers, width, fidelity, populations and fit quality. This
caller does not duplicate its fitter, layout or quality policy. Missing pairs
cannot fit, and numerical failures do not become zero-valued calibrations.
Inspect the returned quality before treating finite values as useful calibration.

This experiment isolates missing preparations and the GE owner's
`GENumericalError` per tone. The owner wraps numerical fitting or classification
failures and retains the original exception in `__cause__`. Mux reports its
reason as `MuxFitFailure` and continues with the remaining tones. Plotting and
other errors propagate unchanged, so a broken diagnostic plot is not reported
as a failed calibration fit. This is this experiment's error policy; it does
not require other experiments to isolate the same failures.

Post-analysis uses the adopted same-source primary without refitting. Caller
must retain that source association; passing another record's primary with the
same indices is not detectable. `MuxGEExp.PostOptions` has a common `default`
and original-index `overrides` of `GEPostAnalyzeOptions`. Radius is None for
automatic selection or a finite nonnegative value in normalized ADC units.
`consider_other` affects automatic-radius selection only. Post returns indexed
`MuxGEPostAnalysis` or retained/numerical failures and publishes `post_i`. The
existing 3x3 g/e/other diagnostic retains its isolated-other row. A returned
diagnostic is not a calibration approval: an infinite condition number or
unresolved clusters makes it unsuitable for calibration.

Both analysis operations allow `source.cfg=None`. Invalid source shapes,
shot/preparation axes, cfg associations and option indices raise before fitting.
Neither operation reads current settings or hardware. Use fresh plots for a
second analysis operation; duplicate `fit_i` or `post_i` names are rejected.
Caller finishes/releases the plots.

## Calling and saving

```python
from zcu_tools.experiment import RunRecord, load_run, save_run
from zcu_lab.v2.mux.singleshot_ge import MuxGEExp

core = MuxGEExp()
result = core.run(config, context=context)
source = RunRecord(config, result, metadata=metadata)
context.scope.stop.raise_if_error()
# Preserve stopped/partial status in caller metadata; do not infer it from IQ.
save_run(source, native_path, spec=MuxGEExp.AXES_SPEC)
restored, device_snapshot = load_run(native_path, spec=MuxGEExp.AXES_SPEC)
primary = core.analyze(restored, core.Options(), plots=analysis_plots)
post = core.post_analyze(restored, primary, core.PostOptions(), plots=analysis_plots)
```

`config` is a validated `MuxGECfg`; `context` owns this operation's hardware,
plots and stop signal. `native_path` is a caller-selected local Path, and
`metadata` contains the caller's run identity, completion and provenance.
Default native persistence requires a cfg. It stores full tones/readout,
original indices, probe preparation and shot coordinates, and complex raw IQ.
Loading restores the historical source; subsequent edits to a current cfg do
not alter it. No current Form or library is consulted. This source contains the
inputs needed for later primary/post proposals, but the core produces no GUI
writeback.

## Physical smoke limits

Use the public `ftf(settings, workers=1)` environment with typed YAML
`readout.drive.kind: mux`, an ADC receiver list and independent drive lines.
The backend is `ftf-segment-v7`; older replay archives are rejected, not adapted.
All receiver IQ at a shot index must come from the same joint-state draw.
Small truncations and short smoke programs do not establish model convergence,
calibrated probe pulses or reliable readout fits. A poor or failed fit remains
visible as quality evidence or a per-tone failure, not a reason to change the
fitter or splice separately simulated shots.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated, ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import StrictInt, model_validator
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import AxesSpec, Experiment, RunRecord
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime import Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import ModularProgramV2, ResetCfg
from zcu_tools.program.v2.modules import (
    ConstWaveform,
    FlatTopWaveform,
    Join,
    MuxReadout,
    MuxReadoutCfg,
    Pulse,
)

from zcu_lab.v2.support.mux import MuxFitFailure
from zcu_lab.v2.support.mux_sweep_cfg import (
    MuxDriveToneCfg,
    require_tone_value,
    validate_participants,
)
from zcu_lab.v2.support.singleshot.plot import draw_ge_scatter

from .analysis import (
    MuxGEAnalysis,
    MuxGEAnalyzeOptions,
    MuxGEPostAnalysis,
    MuxGEPostOptions,
    analyze_ge,
    post_analyze_ge,
)


def _validate_fixed_pulse(pulse: Pulse.Cfg) -> None:
    for name in ("freq", "gain", "phase", "pre_delay", "post_delay"):
        value = getattr(pulse, name)
        if isinstance(value, QickParam) or not math.isfinite(value):
            raise ValueError(f"{name} must be fixed and finite")
    if pulse.pre_delay < 0 or pulse.post_delay < 0:
        raise ValueError("pulse delays must be nonnegative us")
    if isinstance(
        pulse.waveform, (ConstWaveform.Cfg, FlatTopWaveform.Cfg)
    ) and isinstance(pulse.waveform.length, QickParam):
        raise ValueError("pulse length must be fixed")


class MuxGEToneCfg(MuxDriveToneCfg):
    """One named original tone's required probe drive and optional init pulse.

    name is the nonempty capture-generated qubit identity. drive is probe-on;
    init_pulse, when present, runs before it in both preparations on the same
    generator. All pulses are fixed native PulseCfg (MHz, us, degrees, gain),
    validated only when selected. Nonparticipant settings do not run.
    """

    init_pulse: Pulse.Cfg | None = None

    def validate_pulses(self) -> MuxGEToneCfg:
        """Reject swept values or shared-branch generators for every element."""
        _validate_fixed_pulse(self.require_drive())
        if self.init_pulse is not None:
            _validate_fixed_pulse(self.init_pulse)
            if self.init_pulse.ch != self.require_drive().ch:
                raise ValueError("init_pulse must use the tone's drive generator")
        return self


class MuxGECfg(ConfigBase):
    """Complete named tones, ordered participants and shared acquisition controls.

    reset and readout hold the common modules. tones contains one
    MuxGEToneCfg per original readout position, with unique nonempty names.
    participants is a nonempty, duplicate-free list of original tone indices; its order
    determines execution/data rows (Q3/Q1 maps to original indices 2/0).
    Selected drives are required/validated on independent
    generators distinct from readout. shots is positive. Run sets program rounds=1
    and reps=shots; these derived runtime counts are not experiment inputs.
    initial_delay/relax_delay are finite
    nonnegative us. No capture, bindings or writeback destinations are stored.
    """

    initial_delay: Annotated[
        float,
        FieldMeta(
            label="Initial delay", unit="us", description="Delay before the first shot."
        ),
    ] = 1.0
    relax_delay: Annotated[
        float,
        FieldMeta(
            label="Relax delay",
            unit="us",
            decimals=3,
            description="Delay after each shot.",
        ),
    ] = 1.0
    reset: ResetCfg | None = None
    readout: MuxReadoutCfg
    tones: Annotated[list[MuxGEToneCfg], FieldMeta(element_label="name")]
    participants: list[StrictInt]
    shots: StrictInt

    @model_validator(mode="after")
    def validate_acquisition(self) -> MuxGECfg:
        """Reject invalid counts, full-list length or active channel conflicts."""
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        if any(
            not math.isfinite(v) or v < 0
            for v in (self.initial_delay, self.relax_delay)
        ):
            raise ValueError(
                "initial_delay and relax_delay must be finite nonnegative us"
            )
        if len(self.tones) != len(self.readout.tones):
            raise ValueError("tones must match the complete readout tone count")
        validate_participants([tone.name for tone in self.tones], self.participants)
        channels: set[int] = set()
        for index in self.participants:
            tone = self.tones[index]
            require_tone_value(tone.drive, ("tones", index, "drive"))
            tone.validate_pulses()
            if (
                tone.require_drive().ch in channels
                or tone.require_drive().ch == self.readout.gen_ch
            ):
                raise ValueError(
                    f"tones[{index}].drive requires an independent generator"
                )
            channels.add(tone.require_drive().ch)
        return self


@dataclass(frozen=True)
class MuxGEResult:
    """Paired ADC-length-normalized raw complex IQ, not averaged or classified.

    signals is complex128 (K,2,S), original active tone order, probe off/on,
    then shot. shot_indices is int64 arange(S), prepared_states is int64 [0,1]
    (probe off/on, not physical pure-state labels), tone_indices is int64 (K,)
    of original full-readout indices. All tones at a preparation/shot position
    share the same acquisition shot. The two preparations are separate acquires,
    not a four-state joint assignment calibration. Exactly one completed round
    per preparation is retained; missing preparations remain NaN in every tone.
    """

    signals: NDArray[np.complex128]
    shot_indices: NDArray[np.int64]
    prepared_states: NDArray[np.int64]
    tone_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        validate_ge_result(self)


def validate_ge_result(result: MuxGEResult) -> None:
    """Reject malformed source axes/dtypes or broken cross-tone shot availability.

    Require (K,2,S) complex128, K,S>0, unique original nonnegative
    int64 indices, [0,1] preparations and arange(S) shots. Missing preparation
    rows must be wholly NaN across all tones/shots; finite rows stay unfiltered.
    Raises ValueError; no cfg, fitting, mutation or hardware access occurs.
    """
    indices = result.tone_indices
    if (
        indices.ndim != 1
        or indices.dtype != np.int64
        or not indices.size
        or np.any(indices < 0)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError(
            "tone_indices must be unique nonnegative int64 in participant order"
        )
    signals = result.signals
    if (
        signals.dtype != np.complex128
        or signals.ndim != 3
        or signals.shape[:2] != (indices.size, 2)
        or signals.shape[2] == 0
    ):
        raise ValueError("signals must be complex128 (K,2,S), K,S>0")
    if (
        result.shot_indices.dtype != np.int64
        or not np.array_equal(result.shot_indices, np.arange(signals.shape[2]))
        or result.prepared_states.dtype != np.int64
        or not np.array_equal(result.prepared_states, [0, 1])
    ):
        raise ValueError("shot/preparation axes must preserve acquisition order")
    for preparation in range(2):
        row = signals[:, preparation, :]
        if np.isinf(row).any() or not (np.isfinite(row).all() or np.isnan(row).all()):
            raise ValueError("each preparation must be wholly finite or wholly missing")


def _raw_signal(
    prog: ModularProgramV2, _acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    # The builder copies modules: use only the initialized owner from this acquire.
    for module in prog.modules:
        if isinstance(module, MuxReadout) and module.name == "readout":
            iq = module.collect_raw_iq(prog, read_index=0)
            return iq[..., 0] + 1j * iq[..., 1]
    raise ValueError("GE program is missing its named MuxReadout")


def _validate_source(source: RunRecord[MuxGECfg, MuxGEResult]) -> None:
    validate_ge_result(source.result)
    if source.cfg is not None:
        cfg = MuxGECfg.model_validate(source.cfg.model_dump())
        indices = list(cfg.participants)
        if not np.array_equal(indices, source.result.tone_indices) or cfg.shots != len(
            source.result.shot_indices
        ):
            raise ValueError("source cfg active indices/shots do not match raw result")


class MuxGEExp(Experiment[MuxGEResult, MuxGECfg]):
    """Record paired mux IQ clouds and fit each participating qubit's GE separation."""

    Options: ClassVar[type[MuxGEAnalyzeOptions]] = MuxGEAnalyzeOptions
    PostOptions: ClassVar[type[MuxGEPostOptions]] = MuxGEPostOptions

    def analyze(
        self,
        source: RunRecord[MuxGECfg, MuxGEResult],
        options: MuxGEAnalyzeOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxGEAnalysis | MuxFitFailure, ...]:
        """Fit each tone's probe-off/on IQ clouds from the supplied raw source.

        options selects GE fitting settings by original tone index. Return each
        tone's calibration and fit quality, or its failure reason, and draw fit_i
        in plots. Incomplete pairs do not fit. Source cfg may be absent; malformed
        axes, mismatched cfg or invalid options raise before fitting.
        """
        _validate_source(source)
        return analyze_ge(source, options, plots=plots)

    def post_analyze(
        self,
        source: RunRecord[MuxGECfg, MuxGEResult],
        primary: tuple[MuxGEAnalysis | MuxFitFailure, ...],
        options: MuxGEPostOptions,
        *,
        plots: Plots,
    ) -> tuple[MuxGEPostAnalysis | MuxFitFailure, ...]:
        """Estimate per-tone radius and 3x3 confusion diagnostics without refitting.

        primary contains the ordered GE fits from source; options selects radii
        in normalized ADC units. Return diagnostics or each tone's failure reason
        and draw post_i in plots. These are single-qubit diagnostics, not a joint
        assignment matrix. Source cfg may be absent; invalid source, primary or
        options raise.
        """
        _validate_source(source)
        return post_analyze_ge(source, primary, options, plots=plots)

    AXES_SPEC: ClassVar[AxesSpec[MuxGEResult, MuxGECfg]] = AxesSpec(
        axes=(
            AxesSpec.Axis("shot_indices", "Shot Index", "", dtype=np.int64),
            AxesSpec.Axis("prepared_states", "Probe On", "", dtype=np.int64),
            AxesSpec.Axis("tone_indices", "Tone", "", dtype=np.int64),
        ),
        z=AxesSpec.Z("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=MuxGEResult,
        cfg_type=MuxGECfg,
        cfg_schema_version="3.0",
        tag="mux/singleshot/ge",
    )

    def run(self, config: MuxGECfg, *, context: RunContext) -> MuxGEResult:
        """Measure paired probe-off/on IQ shots for config's ordered participants.

        Each preparation runs common reset, simultaneous optional init pulses
        followed by probes (on only for the second preparation), then mux readout
        after the longest branch. Return normalized raw IQ shaped (K,2,shots),
        without averaging, sorting or classification. K follows participants;
        the two rows mean probe off/on, not guaranteed pure ground/excited states.
        Use one round with reps=shots, leaving config unchanged.
        context supplies hardware, stop signal and live IQ scatter plots. Missing
        preparations remain NaN; check context.scope.stop.raise_if_error()
        before reporting success. Invalid config or program builds raise.
        """
        # 1. Use one raw-shot round and preserve the requested participant order.
        cfg = MuxGECfg.model_validate(config.model_dump())

        indices = list(cfg.participants)
        readout = cfg.readout.with_updates(
            tones=[cfg.readout.tones[i] for i in indices]
        )
        shots = np.arange(cfg.shots, dtype=np.int64)

        # 2. Show both preparations without averaging away their IQ spread.
        def update(data: NDArray[np.complex128]) -> None:
            for row, index in enumerate(indices):
                draw_ge_scatter(
                    context.plots,
                    f"measurement_{index}",
                    data[:, :, row],
                    title=f"{index}: {cfg.tones[index].name}",
                )

        buffer = SignalBuffer((2, cfg.shots, len(indices)), on_update=update)
        with Schedule(cfg, buffer, scope=context.scope) as sched:
            for with_probe, step in sched.scan("probe on", [False, True]):
                # 3. Start tone branches together; read out after the longest one.
                # Init runs in both preparations, probe only in the second.
                branches = [
                    [
                        Pulse(f"init_{i}", cfg.tones[i].init_pulse),
                        Pulse(
                            f"probe_{i}",
                            cfg.tones[i].require_drive() if with_probe else None,
                        ),
                    ]
                    for i in indices
                ]
                builder = (
                    step.prog_builder(
                        context.soc,
                        context.soccfg,
                        reps=cfg.shots,
                        initial_delay=cfg.initial_delay,
                        relax_delay=cfg.relax_delay,
                    )
                    .add_reset("reset", cfg.reset)
                    .add(Join(*branches), MuxReadout("readout", readout))
                )
                prog = builder.build()
                # Read normalized shots from this program's initialized readout.
                _ = builder.run_program(
                    prog, raw2signal_fn=partial(_raw_signal, prog), rounds=1
                )

        # 4. Keep completed preparations in participant/preparation/shot order.
        buffer.trigger_update(flush=True)
        return MuxGEResult(
            buffer.array.transpose(2, 0, 1).copy(),
            shots,
            np.array([0, 1], dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
        )
