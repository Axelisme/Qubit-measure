# Mux Ramsey core

`MuxT2RamseyExp.run(config, *, context)` measures selected qubits in one program
with independent x90 drives and a common mux readout. It has no Form, recipe,
catalog registration or writeback proposal.

## Configuration and timing

`participants` is a nonempty, unique list of names in the complete `tones` list.
Q3/Q1 selects original indices `[2, 0]`, retaining that execution/data order.
Every element requires valid drives on independent generators, even when not
participating. Names, count and order match the fixed capture.

Import `MuxT2RamseyCfg`, `MuxT2RamseyExp` and `MuxCoherenceToneCfg` from
`zcu_lab.v2.mux.t2ramsey`. The cfg contains:

- `modules: MuxDriveModuleCfg`, with optional common reset and full mux readout.
- `tones: list[MuxCoherenceToneCfg]`, one element per original readout tone.
  Each element has a fixed `name`, required `drive` and `detune` in MHz.
  `drive` is the source x90 pulse, not x180. All drives are required, even for nonparticipants.
- `delay: SweepCfg`, one common requested free-evolution delay in us.
- Shared `reps`, `rounds`, `initial_delay` and `relax_delay`, with times in us.

Participants must be nonempty. All drives must use independent generators,
different from the readout generator. Pulse parameters must be fixed and finite.
The delay bounds must be finite and nonnegative. Numeric cfg validation does not
resolve component identities; the caller supplies valid tone/qubit bindings.

The sequence is reset, Join of first x90 pulses, common Delay, Join of final x90
pulses, then active mux readout. Each Join starts its branches at the same stage
origin. Native pulse pre/post delays remain unchanged. The Delay starts after the
longest first stage, not independently after each qubit's own pulse.
For zero pre/post delays, a shorter pulse waits an extra fixed
`max_len - len_i`. This can change fringe amplitude and constant phase.
The analysis still uses the common requested delay. Enable free fit phase when
a constant phase offset matters. Native clock differences of a few ticks remain;
the core does not end-align, pad, double-compile or save compiled actual delays.

Each final x90 phase uses its source phase plus its own nominal detune times the
shared delay parameter. Program/sweep owners perform native conversion.
`actual_detune(..., time_scaler=1.0)` owns the realized MHz detune estimate.
The estimate preserves that owner's phase-wrap convention, including for signed
detunes; do not treat it as a signed unwrapped frequency correction without
checking the acquisition range.

## Run and analysis

```python
from zcu_tools.experiment.records import RunRecord
from zcu_lab.v2.mux.t2ramsey import MuxT2RamseyExp

core = MuxT2RamseyExp()
result = core.run(cfg, context=context)
context.cancel_signal.raise_if_error()
source = RunRecord(cfg=cfg, result=result)
outcomes = core.analyze(source, core.Options(fit_phase=True), plots=analysis_plots)
```

RunContext supplies compatible soc/soccfg, a fresh StopSignal and caller-owned
Plots. The caller finishes/releases plots and cleans up devices. Run does not
save files or cache a previous result.

`MuxCoherenceResult` contains original active `tone_indices (K,)`, common requested
`times (P,)` in us, `signals (K,P)` as ADC-length-normalized complex average IQ,
`actual_detunes (K,)` and `source_frequencies (K,)` in MHz. Source frequencies are
the supplied x90 carriers before the phase ramp; full pulse settings remain in cfg.
Actual detunes may be NaN when a nonzero nominal detune has no positive rounded
delay. The core preserves the owner's diagnostics and NaN, without substituting a
nominal rate. Save/load and analysis still accept this source.

The hardware sweeps delay and averages reps/rounds. Cancellation keeps the mean of
completed rounds across the whole sweep. With no completed round, IQ stays NaN.
Call `raise_if_error()` before declaring success, then inspect `is_set()` to
distinguish cancellation. Acquisition errors preserve their cause even with
partial data. Build, validation and conversion errors propagate directly.

Analysis accepts explicit RunRecord, including cfg=None. `fit_fringe=True`
selects the existing damped-cosine fitter; False selects exponential decay.
`fit_phase=False` uses the single-qubit nearest-zero 0/180-degree phase policy.
True frees fringe phase. Decay ignores fit_phase. Missing IQ is filtered per tone;
a fringe fit needs six points and a decay fit four.

Each original index returns `MuxCoherenceFit` with T2* and its error in us,
fringe rate and its error in MHz, selected model and FitQuality. Numeric failures
return `MuxFitFailure(index, reason)` without losing other tones. Source contract
errors raise instead of becoming fit failures. Graphs are named
`measurement_i` and `fit_i`, so duplicate labels do not overwrite them.
A finite fit is not a calibration-quality guarantee.

## Native source persistence

`save_run(source, path, metadata=...)` stores full cfg, inactive tones, original
indices, shared requested delay, IQ, actual detunes and source frequencies.
Time and frequency use seconds and Hz on disk, us and MHz in memory.
`load_run(path)` restores an explicit source; changing a current cfg does not
retarget or alter its analysis. The caller sets complete/partial/stopped metadata
from the run outcome, not from the existence of IQ.

This core produces no proposal. A later calibration caller must require a usable
fringe fit, source frequency and actual detune before considering a Ramsey
frequency correction. NaN detune does not meet that requirement.

## Adapter source and proposals

`MuxT2RamseyAdapter` in `gui.py` uses the explicit Run/Load record and its capture.
It is not registered in the catalog yet; this leaf has no generated Form or
recipe entry. The adapter alone is not a GUI launch path.

Summary rows include original tone `position`, source `name`, selected fit model,
quality and values, or a failure reason. Q3/Q1 retains indices 2/0 in that row
order. Missing source cfg leaves the name unavailable; current settings do not
supply it. Quality errors point to the corresponding summary row.

Each successful, fully finite tone proposes `<qubit>.t2r` in us and a T2R sample
at captured bias. A frequency candidate additionally requires source cfg, a
fringe fit, finite source carrier, finite actual detune and finite fitted fringe
rate. It proposes source carrier + actual detune - fitted fringe rate in MHz.
NaN actual detune suppresses only this correction; it is never replaced by zero
or nominal detune. The usable coherence result remains available.

Destinations come only from the same Run/Load capture. Missing capture produces
no proposals; inconsistent captured bindings raise. Failed or partial rows
produce none. Inspect phase/rate conventions, quality, uncertainty and destination
before explicitly applying individual items. Analysis and proposal creation do
not write.
