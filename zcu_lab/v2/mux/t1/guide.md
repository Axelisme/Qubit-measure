# Mux T1

## GUI setup

Open `mux/t1` from **New Tab** after binding `$mux` to a MuxGroup.
The capture must contain every qubit and its matching resonator in readout order.
Readout references `$mux.module.readout` when present, preserving the library's tones and scalar settings.
The stored module must use the mux family with the full group tone count.
Without a slot, Readout starts inline from the blank mux template with group calibration tones; complete its required fields before running.

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

## Adapter source and proposals

`MuxT1Adapter` in `gui.py` connects the core to explicit Run/Load sources.
Its native generated Form is registered as `mux/t1`.
The registered MCP recipe is `mux_t1`. See the
[mux recipe guide](../../../recipes/mux-guide.md#mux_t1)
for typed overrides, origins, source-bound reporting and explicit writeback decisions.

Summary rows retain each original tone `position`, source `name`, fitted values and
quality, or a failure reason. Q3/Q1 gives indices 2/0 in that row order. Quality
errors identify the summary row. Missing source cfg leaves the name unavailable;
the adapter does not reconstruct it from current settings.

Each successful, fully finite row proposes its captured qubit's `t1` in us with
its fitted error, plus a T1 sample at captured bias. The destination comes only
from the same Run/Load capture's full binding. Missing capture produces no
proposals; inconsistent captured bindings raise. Failed and partial rows produce
none. Inspect the decay model, quality, uncertainty and destination before Apply.
Analyze and proposal creation do not write or apply candidates.
