# Constructive relative phase

**Readers:** operator, agent. Run `two_qubit/constructive_phase` and interpret its count-derived return and phase proposals.

## Sequence and calibration

Scan Qubit2 minus Qubit1 relative phase in degrees and common envelope duration in us.
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

Bind `qubit1` and `qubit2` to distinct qubits and `mux` to a group containing both.
Bind `coupler` to the component that stores their CZ pulses.
Notebook uses the same built-in position names, but does not bind the GUI aliases.
For an existing point, bind the new aliases. Rename both CZ slots and their library keys to
`cz_qubit1` and `cz_qubit2`. Make each slot point to its renamed key.
Open the tab or choose Defaults to set the hidden tone indices from `$mux.qubits`.
Check both Xπ pulses and all mux tones, ADC routes and scalar timing.
Readout references `$mux.module.readout` when present. Otherwise fill the blank
mux template, including classifiers. Missing sources remain required inputs.

The coupler's `cz_qubit1` and `cz_qubit2` slots seed editable ordinary charge pulses.
Without slots, fill both cosine templates, including signed gain, frequency,
envelope length and Nyquist zone. Channels come from captured qubit wiring.
Only Qubit1/Qubit2 charge lines play the drive; the coupler stays DC-biased.
Both source frequencies, envelope lengths and pre-delays must match.
The sequence waits for both drive pulses, including their individual post-delays.
Keep the signed gain ratio fixed.
Each phase coordinate sets Qubit2.phase to saved Qubit1.phase plus that coordinate.
The common frequency and Qubit1 phase remain fixed. Choose the transition using
conditional spectroscopy and the duration window using chevron.
Only common envelope duration changes within each trace.

The Form lists reset, Xπ preparation pulses, dual-line drive and readout in
playback order. Set the outer relative phase
sweep before the inner duration sweep. Repetitions, rounds and acquisition
delays appear last.
Enter finite scan endpoints and strictly positive durations.
Review repetitions, rounds, optional reset and timing before Run.
Refresh keeps authored inputs and rechecks the [mux tone count](../guide.md#綁定位置與讀出-tones) against the new capture.
Use Defaults to replace unsaved input with initial values from the new source.

## Read the figures

Analyze returns only `relative_phase_deg`, `reason` and `interference_quality`.
Per-phase rates, populations and interference curves stay inside analysis for plotting.

Use the live and Analyze interference maps to locate the fastest oscillation.
Both show terminal P(g,g) after the final double Xπ, divided by all nine
terminal class counts. The live title shows acquisition progress. Use the
return figure to check classification at one phase. It shows three fractions:
both qubits in g, both in e, and
anything else (the other seven joint outcomes). They add up to one at each
duration. It shows the measured phase closest to the proposed relative phase,
or the first measured phase when no proposal is available. The title gives the
plotted phase in degrees. The grouped curve cannot identify which of its
seven individual outcomes contributes most.
These figures include preparation and readout errors.
They do not measure gate fidelity or resolve physical leakage by themselves.
There are no mean-IQ diagnostics or saved IQ clouds.

Each fully acquired duration trace with nonzero selected counts uses an
undamped cosine fit.
Rates below 1 divided by the saved duration span are unresolved.
Duration-fit rates distinguish unresolved fitted rates from identifiable rates.
A failed or degenerate fit shows a reason, not a zero rate. Use Duration slices
to compare measured duration points with the dense cosine fit.
The duration points and the grouped terminal probabilities show ±1 shot SEM
from the corresponding class indicator and that point's retained count.
This is counting noise, not the uncertainty of the fitted rate. The 2D map
has no per-cell bars.
Choose spacing so the expected rate is below `1/(2 × duration spacing)`.
Check aliasing manually. Thermal mixtures may contain more than one rate.

Identifiable rates use `rate² = A + B cosφ + C sinφ`.
The proposed relative phase is `atan2(C,B)` in degrees, the maximum-rate branch.
More than three identifiable phase slices and full-rank phase coverage are
needed for finite uncertainty. Zero modulation or nonfinite uncertainty prevents
a proposal. The interference plot shows the dense rate-squared fit and its
maximum. R² is the fraction of measured rate-squared variation explained by
the fit; it has no additional selection threshold. This phase is not a logical
local-Z correction.

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

Load leaves the current Form unchanged. Analysis uses the saved cfg (schema 5.0) and counts.
Native Load rejects older results whose axis names do not match this layout. Keep the original file and use the matching older version to read it.
If the result layout matches but the cfg major does not, Load retains Result and metadata without cfg. Fitting requires cfg.

## Review and apply phase proposals

FIT proposes only Qubit2's ordinary charge pulse and its independent coupler slot.
The targets are `<coupler>.pulses.cz_qubit2` and `<coupler>.module.cz_qubit2`.
The original Run or Load capture supplies the coupler, even if you change aliases afterward.
Writeback fails without that capture.
Qubit2.phase becomes saved Qubit1.phase plus the fitted relative phase.
Qubit2 frequency, signed gain, waveform and timing stay as measured.
Qubit1 stays the phase reference. Missing cfg or an unresolved phase gives no proposal.
Inspect the captured destination and waveform before Apply.
The pulse and slot rows apply independently; review partial selections.
Only an applied proposal becomes calibration for later Forms.

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

Use the adapter with a complete saved or operator-reviewed cfg.
`point_view` is the bound ParameterView for the save destination.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.constructive_phase.core import ConstructivePhaseExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(ConstructivePhaseExp())
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
