# Mux Echo core

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
`T2EchoAnalyzeOptions`: fit_method defaults to decay, or select fringe explicitly.
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
Disk units are seconds/Hz; memory units are us/MHz. `load_run(path)` restores the
source without reading a current cfg or Form. The caller sets
complete/partial/stopped from the run outcome.

This core makes no calibration proposal. A later caller can use a usable Echo
fit for T2E, independently of unavailable detune. NaN actual detune must never be
replaced by a guessed frequency correction.
