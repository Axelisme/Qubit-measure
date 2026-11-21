# Rabi chevron

**Readers:** operator, agent. Run `two_qubit/chevron` and interpret its count-derived return.

## Sequence and calibration

Scan common drive frequency in MHz and common envelope duration in us.
Each shot emits optional reset, nominal B11 preparation, simultaneous two-line
drive, double Xπ contrast and one terminal mux readout.
`reset=None` emits no reset. Set a reset cfg only after reviewing its pulse
and timing against the authorized limits. The Defaults button recaptures
inputs; it does not emit a reset pulse.

Both qubit tones require circle classifiers with g/e centers and radius.
Run mux singleshot GE Post under these readout conditions before Run.
Missing classifiers, linear classifiers and missing radii make the cfg invalid.
Classification happens during acquisition, not during analysis.
The three classes are g, e and other. Other stays in the denominator.

## Prepare the Form

In the point's Aliases table, bind `$qubit1` and `$qubit2` to distinct components.
Bind `$mux` to their mux group and `$coupler` to the component that stores
the pair's CZ pulses. Rebind older points in the same table. GUI aliases
and Notebook's built-in roles of the same names are separate.
Opening the tab or choosing Defaults uses `$mux.qubits` to set two hidden tone indices.
If the capture has no `$coupler` binding, the form reports it before Run.
Check both Xπ pulses and all mux tones, ADC routes and scalar timing.
Readout references `$mux.module.readout` when present. Otherwise fill the blank
mux template, including classifiers. Missing sources remain required inputs.

The coupler's `cz_qubit1` and `cz_qubit2` slots seed editable charge pulses
in `drive.qubit1` and `drive.qubit2`. On an older point, set these slot names
and use the same new names for their pulse library keys.
If a slot is missing, enter signed gain, envelope length and Nyquist zone in
the cosine template. The channel comes from that qubit's captured wiring.
The common scan sets both frequencies.
Only Qubit1/Qubit2 charge lines play the drive; the coupler stays DC-biased.
Both source frequencies, envelope lengths and pre-delays must match.
The sequence waits for both drive pulses, including their individual post-delays.
Keep the signed gain ratio fixed.
At each frequency coordinate, both frequencies change together.
Individual phases remain fixed; choose relative phase with constructive_phase first.
Only common envelope duration changes within each trace.

The Form lists reset, Xπ preparation pulses, dual-line drive and readout in
playback order. Set the outer common frequency sweep before the inner
duration sweep. Repetitions, rounds and acquisition
delays appear last.
Enter finite scan endpoints and strictly positive durations.
Review repetitions, rounds, optional reset and timing before Run.
Refresh keeps authored inputs and rechecks the [mux tone count](../guide.md#綁定位置與讀出-tones) against the new capture.
Use Defaults to replace unsaved input with initial values from the new source.

## Read the figures

Analyze publishes these figures and returns `None`. Per-frequency rates, counts
and curves stay inside analysis; frequency and duration selection remain manual.

The live and Analyze chevron maps show P00 after double Xπ contrast.
P00 is the terminal (g,g) count divided by all nine terminal class counts.
The return figure shows all nine g/e/other joint outcomes.
These figures include preparation and readout errors.
They do not measure gate fidelity or resolve physical leakage by themselves.
There are no mean-IQ diagnostics or saved IQ clouds.

Each fully acquired duration trace with nonzero selected counts uses an
undamped cosine fit.
Rates below 1 divided by the saved duration span are unresolved.
Failed or degenerate fits report a reason rather than a zero rate.
Compare the measured return traces with the fitted curves.
Choose spacing so the expected rate is below `1/(2 × duration spacing)`.
Check aliasing manually. Thermal mixtures may contain more than one rate.

Frequency and duration selection remain manual. A valley only identifies a
window for later amplitude and phase calibration.

## Save and load

Native save keeps the complete source cfg, coordinate and duration axes, and
float64 `sums` with shape `(Nc,Nd,3,3)`.
Class axes are terminal Qubit1, terminal Qubit2.
Every class axis is ordered g, e, other.
Rounds accumulate during acquisition without normalization.
File size does not grow with repetitions or rounds.
Completed rounds survive cancellation; unmeasured points remain NaN.
A point with counts from at least one completed round can enter the fit if it has accepted shots.
Fits require all requested shots at every point of a duration trace.
Always check the execution status before accepting partial results.

Cfg schema is 5.0. Load leaves the current form unchanged, and analysis uses the
saved cfg and counts. The current Load rejects older files whose axis names no
longer match. Use the matching older version to read the original file. No cfg
or file conversion runs on Load.

## Select shots with mux herald

Choose `reset/mux_herald` in Reset. Use the same mux readout as the terminal
readout. Set `keep: {'0': g, '1': g}` if Qubit1 and Qubit2 occupy positions 0 and 1.
Use their original tone positions from the full readout, not their ADC numbers.

Probabilities are conditional on passing herald. Divide each terminal class
count by the point's total count. Acceptance is the total divided by
reps × rounds, not ground-state purity. A point with no accepted shots has
NaN probabilities. Completed rounds remain usable when Run stops early.
Files saved under the former separate heralded experiment no longer load.
After amplitude/frequency selection, measure conditional phase and local Z.
Repeat calibration after changing pulse shapes or the pair's working point.
Validate the actual FTF control chain separately from offline compile checks.

## Notebook

Supply a complete cfg. Set `qubit1_tone` and `qubit2_tone` to zero-based
positions in the full readout.
`point_view` is the bound ParameterView for the save destination.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.chevron.core import ChevronExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(ChevronExp())
record = exp.run(cfg)
exp.analyze(None)
exp.save_run(point_view, source=record)
```

The coordinate is a hardware sweep. Constant and flat-top pulse durations also
use a hardware sweep in the same program. Sampled cosine, Gaussian and DRAG
waveforms reject swept lengths, so only their duration uses a host loop.

Fits use count-weighted binomial errors. Endpoint probabilities use
Jeffreys-smoothed variance so observations at zero or one retain finite weight.

## Calibration order and field units

Calibrate each qubit's Xπ slot with amplitude and length Rabi before using it.
Confirm pulse frequency in MHz, signed gain in [-1, 1], and duration in us.
Use conditional spectroscopy to choose the charge-line frequency.
The operator must confirm the common duration range in us before Run.
