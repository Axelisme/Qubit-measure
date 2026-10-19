# Mux Echo

## GUI setup

Open `mux/t2echo` from **New Tab** after binding `$mux` to a MuxGroup.
The capture must contain every qubit and its matching resonator in readout order.
An existing readout slot must use the mux family with that full tone count.
If no readout is stored, fill the blank mux readout before running.

The Tones picker chooses which element to edit, not which qubit runs.
Name is read-only. Ordinary edits cannot add, remove, rename or reorder tones.
Set Participants to a JSON name list, for example `["Q3", "Q1"]`.
That order becomes execution/data rows for original positions 2/0.
The picker can show Q2 while Q3/Q1 run.
Empty, duplicate and unknown participants remain editable invalid drafts and block Run.

Every tone needs a valid drive, including nonparticipants.
Errors in another picker element remain visible on the Tones row.
Select that element to fix them; hiding it or removing its name from Participants
does not bypass validation. Reps, rounds and recovery delay are common controls.
Recovery initially uses five times the largest captured T1, with the existing
missing-T1 default. A stored invalid T1 leaves recovery blank for correction.

## GUI results and saved sources

Run draws a combined `measurement` figure.
All lines use the common requested-delay axis.
The legend uses original index and source cfg name in participants order.
Per-tone native `measurement_i` figures remain available.
Analyze reads the selected Run/Load source, not the current Form.
Use the Analysis figure picker to inspect `fit_i`.
Here `i` is an original tone position, never a participants row number.

Use Data to save measurement data and analysis artifacts.
Load Data selects a native file or a saved run.
Loading replaces the analysis source without changing current Form inputs.
A later Form edit does not alter the historical pulse, binding or destination.
Failed or partial rows do not produce calibration candidates.
Read each Writeback preview destination, value and fit diagnostics.
Deselect unwanted items before **Apply Selected**.
Preview and Analyze do not write; applying one item does not apply other candidates.

## Form drafts

Form persistence contains exactly `data` and `selections`.
Selections use typed paths with integer list positions.
Invalid leaf text and participants drafts survive save/restore.
Restoration reports dropped fields and does not run converters or refresh capture.
If the saved input cannot be read, the host reports it and uses fresh defaults.
Reset explicitly captures current parameters and rebuilds initial values.

A raw `FormState.write` replaces the complete value at its path and clears selections
at or below it. To change one leaf, use its complete path, for example
`("tones", 0, "drive", "module", "gain")` for an inline drive.
Do not send a partial tone or pulse dictionary expecting a merge.
Writing does not run a converter; selecting another module kind does.
Recipe/MCP logical field paths omit the inline wrapper, for example
`("tones", 0, "drive", "gain")`. The integer is the full original tone position,
not a row in Participants and not a name key.

## Core reference

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
Disk units are seconds/Hz; memory units are us/MHz.
The native file is loadable when the additional Labber export fails: IQ uses a
delay/tone grid, while detune and source frequency use only the tone grid.
The save report marks this as `Partially saved`, with native `saved` and Labber
`failed`; it does not report an all-formats success. `load_run(path)` restores the
source without reading a current cfg or Form. The caller sets
complete/partial/stopped from the run outcome.

This core makes no calibration proposal. A later caller can use a usable Echo
fit for T2E, independently of unavailable detune. NaN actual detune must never be
replaced by a guessed frequency correction.

## Adapter source and proposals

`MuxT2EchoAdapter` in `gui.py` uses the explicit Run/Load record and its capture.
Its native generated Form is registered as `mux/t2echo`.
The registered MCP recipe is `mux_t2echo`. See the
[mux recipe guide](../../../recipes/mux-guide.md) for typed overrides, origins,
source-bound reporting and explicit writeback decisions.

Summary rows include original tone `position`, source `name`, fit model, quality
and values, or a failure reason. Q3/Q1 gives indices 2/0 in that row order.
Quality errors identify the summary row. Missing source cfg leaves the name
unavailable; the adapter does not read current settings to reconstruct it.

A successful, fully finite tone proposes its captured qubit's `t2e` in us and
a T2E sample at captured bias. Echo does not propose a frequency correction.
Unavailable actual detune does not suppress an otherwise usable T2E fit.
Missing capture produces no proposals; inconsistent captured bindings raise.
Failed and partial rows produce none. Inspect the model, quality, uncertainty
and captured destination before explicitly applying individual items.
Analyze and proposal creation do not write.
