# CZ common-amplitude calibration

**Readers:** operator, agent. Run common-amplitude CZ calibration and review first-return pulse proposals.

## Choose the experiment

Open `two_qubit/cz_amplitude` for a return scan. Leave `reset` unset to omit reset.

Bind `qubit1` and `qubit2` to distinct qubits and `mux` to a group containing both.
Bind `coupler` to the component that stores their CZ pulses.
Notebook uses the same built-in position names, but does not bind the GUI aliases.
For an existing point, bind the new aliases. Rename both CZ slots and their library keys to
`cz_qubit1` and `cz_qubit2`. Make each slot point to its renamed key.
Opening the tab or choosing Defaults captures aliases, parameters and library once. Refresh keeps the draft and checks the [mux tone count](../guide.md#綁定位置與讀出-tones) against the new capture.
After changing bindings or calibrations, choose Defaults and check the whole Form again.

## Preparation and inputs

Supply calibrated `$qubit1.module.x180`, `$qubit2.module.x180`, and the complete mux
readout. Readout references `$mux.module.readout` when present.
Without that slot, fill every required field in the blank mux template.
Keep all tones in their original order. Opening the tab or choosing Defaults sets hidden tone indices from `$mux.qubits`.

Both qubit tones require circle classifiers with radius. Missing classifiers,
linear classifiers and missing radii make the cfg invalid before Run.
Run GE Post under the same readout conditions first.
Classification produces g, e and other. Other stays in the probability denominator.

The two ordinary CZ pulses start from `$coupler.module.cz_qubit1` and `cz_qubit2` when
available. Without those slots, enter frequency, signed gain, Nyquist zone and
cosine duration. Generator fields come from captured wiring.
Both pulses need matching frequency, duration and pre-delay on independent
generators. Their post-delays may differ; the sequence waits for both.
Only the two charge lines are driven. The coupler stays DC-biased.

The Form lists Reset, Qubit1/Qubit2 Xπ, Two-line CZ drive and Mux readout
before the sweep and acquisition fields. Use the calibrated pulse library
entries or inspect the Inline fields. Gen ch is the wired generator channel.
Frequency is the RF carrier; Mixer freq is the digital mixer offset, if set.
Keep NQZ, the Nyquist zone, from the source pulse or confirm it for the wired
channel. Waveform Length is the envelope duration, not the delays. Phase sets
the carrier angle. Pre-delay and Post-delay place the pulse in the sequence.
Keep the relative CZ phase when varying the scale.

For Common CZ gain multiplier, enter nonnegative Start and Stop and choose
Expts, the number of points including both endpoints. The scan multiplies
both signed gains by each scale. It does not change RF, relative phase or
waveform length. Reps is the attempted shot count at each scale per round;
Rounds repeats and accumulates complete sweeps. Initial delay waits before
the first shot and Relaxation delay waits after each shot.

## Run and inspect

The sequence is `optional reset → paired Xπ preparation → simultaneous CZ
→ paired Xπ contrast → terminal mux`. The preparation is called B(11): both
qubits receive Xπ, so both flip if they start in g. In each paired Xπ block,
the two pulse timelines start together and the sequence waits for both to end.
A pulse's own pre-delay can shift its envelope within the block. Without
reset, thermal population remains in the ensemble. This is a gain scan,
not a frequency scan.

The live figure shows terminal P(g,g), the number of scale points with data,
and the accumulated retained shots per point. Full point coverage can occur
before all requested rounds have run. Without herald or cancellation, the
final retained count per point equals Reps × Rounds. With herald, compare the
retained count with that attempted-shot budget; they need not match.
P(g,g) divides the terminal (g,g) count by the total of all nine classes.
Analyze shows the measured P(g,g) points and the first-return fit on one
figure. The joint-outcome panels show all nine terminal probabilities over
g, e and other; use them to tell which outcomes gain population as
P(g,g) falls. Qubit1 is the first entry in P(a,b). Five outcomes contain
`other`: two in the g/e panels and three in the other panel. A point that
looks like zero at this scale is not proof of zero counts or zero leakage.
Each measured probability bar is ±1 shot SEM from the retained classified
shots at that scale: √[p(1−p)/(N−1)] for at least two retained shots. For
mux herald, N counts only the shots that passed. The fit uses Jeffreys-smoothed weights at endpoints; these
weights are not the plotted bars. Period standard error is a separate fit
uncertainty.
No per-shot IQ or mean-IQ diagnostic is stored or plotted.

The return is a leakage proxy, not physical gate fidelity.
Thermal population, preparation and readout errors remain in it.
Other denotes classifier rejection; it is not a calibrated higher-level population.
A high return alone does not establish a π conditional phase.

## Fit the first complete return

Analyze returns `fit`, the first-return fit summary.
It contains `scale`, `period`, `error`, `offset`, `contrast`, `fit_quality` and
`reason`. Counts, probabilities and curves stay inside analysis
for plotting.

FIT sends P00 to the fixed-zero-phase model
`offset + contrast cos(2π scale/period)`. The candidate common scale is
the measured first positive return, `period`.

Measure scale=0 and continue through at least one full cycle.
FIT uses the contiguous prefix with all requested shots acquired.
It stops that prefix at the first unmeasured scale or scale with zero accepted shots.
A cancelled suffix after a full first cycle still allows a proposal.
Partial scales remain visible but do not enter the fit.
Use enough points to resolve the oscillation; sparse sampling can alias a faster one.

The figure overlays the fit and labels the candidate or failure reason.
The summary includes offset, contrast, period, period standard error and
fit quality. R² is the fraction of the measured return variation explained
by the fit. Normalized RMS is the root mean square of the fit residuals
divided by the observed probability span. Each relative fit SE is that
parameter's standard error divided by its absolute fitted value. These
metrics have no separate numeric selection threshold.
Numeric failure, zero contrast, a degenerate model,
nonfinite uncertainty or incomplete coverage stops the proposal.

Finite identifiable fits remain eligible with low nonzero contrast.
Inspect the measured curve and uncertainty before accepting a proposal.
Thermal mixtures and large-gain AC Stark shifts can violate the single-frequency,
fixed-phase model.

## Preview and selected Apply

Both versions propose the same four independent rows:

| Target | Proposal |
| --- | --- |
| `$coupler.pulses.cz_qubit1` | Source Qubit1 pulse with gain multiplied by the fitted scale. |
| `$coupler.module.cz_qubit1` | Concrete key for that pulse in the Run/Load capture. |
| `$coupler.pulses.cz_qubit2` | Source Qubit2 pulse with gain multiplied by the same scale. |
| `$coupler.module.cz_qubit2` | Concrete key for that pulse in the same capture. |

Only the gains change. Preview the destination, generator, signed gain,
frequency, relative phase and waveform. Pulses come from the saved run cfg.
The original Run or Load capture supplies the coupler, even if you change aliases afterward.
Writeback fails without that capture.

Select the rows you accept, then press Apply Selected.
Pulse rows and slot rows apply independently. Apply each required slot explicitly.
Choose Defaults in other open forms to capture accepted calibrations.
Alternate amplitude and conditional-phase/frequency manually.
After changing pulse shape, duration, phase, gain, frequency or bias,
recheck CZ and remeasure both local-Z corrections.

## Save, load and partial data

Save stores the source cfg and float64 `sums` with shape `Ns × 3 × 3`.
Class axes are terminal Qubit1 then terminal Qubit2, each g, e, other.
Counts remain unnormalized. Storage does not grow with the shot count.
Unmeasured scales are NaN. Cancellation retains completed rounds.

Native Load restores the saved counts and cfg without changing current Form inputs.
The cfg schema is 6.0. Native Load rejects older results whose axis names do not match this layout.
Keep the original file and use the matching older version to read it.
If the result layout matches but the cfg major does not, Load retains Result and metadata without cfg. Fitting requires cfg.
Analysis never looks up current calibrations. Missing saved cfg prevents the fit.
Old per-shot IQ files without sums are rejected; there is no automatic conversion.
`stop_requested` records a stop request, not data completeness.
Acquisition exceptions propagate unchanged from Run. A failed Run publishes no
new record and leaves the previous successful record unchanged, even after Stop.

## Select shots with mux herald

Choose `reset/mux_herald` in Reset. Use the same mux readout as the terminal
readout. Set `keep: {'0': g, '1': g}` if Qubit1 and Qubit2 occupy positions 0 and 1.
Use their original tone positions from the full readout, not their ADC numbers.

Probabilities are conditional on passing herald. Divide each terminal class
count by the point's total count. Acceptance is the total divided by
reps × rounds, not ground-state purity. A point with no accepted shots has
NaN probabilities. Completed rounds remain usable when Run stops early.
Files saved under the former separate heralded experiment no longer load.

## Notebook

Use the adapter with a complete saved or operator-reviewed cfg.
`point_view` is the bound ParameterView for the save destination.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.cz_amplitude.core import CZAmplitudeExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(CZAmplitudeExp())
record = exp.run(cfg)
exp.analyze(None)
exp.save_run(point_view, source=record)
```

The common gain scale is a hardware sweep in one compiled program.

Fits use count-weighted binomial errors. Endpoint probabilities use
Jeffreys-smoothed variance so observations at zero or one retain finite weight.

## Calibration order and field units

Calibrate each qubit's Xπ slot with amplitude and length Rabi before using it.
Confirm pulse frequency in MHz, signed gain in [-1, 1], and duration in us.
Use lookback to set mux timing, then mux onetone to set frequencies and gains.
Confirm readout timing in us and gains in [-1, 1], then repeat GE Post under those conditions.
Choose the dual-line CZ pulse with chevron and confirm the coupler's `cz_qubit1` and `cz_qubit2` slots.
The operator must confirm the common gain-scale range before Run.
