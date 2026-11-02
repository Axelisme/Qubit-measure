# CZ common-amplitude calibration

**Readers:** operator, agent. Run common-amplitude CZ calibration and review first-return pulse proposals.

## Choose the experiment

Open `two_qubit/cz_amplitude` for an unconditioned return scan.
Open `two_qubit/cz_amplitude_heralded` to condition the analysis on a preceding
herald classified as ground for both qubits. Both versions accept optional
`reset`. Leave it null to emit no reset.

Bind `f1` and `f2` to distinct fluxoniums, `coupler` to their transmon,
and `mux` to a MuxGroup containing both qubits. Open or Reset captures aliases,
parameters and library once. Refresh preserves authored inputs.
After changing bindings or calibrations, Reset and check the whole Form again.

## Preparation and inputs

Supply calibrated `$f1.module.x180`, `$f2.module.x180`, and the complete mux
readout. Readout references `$mux.module.readout` when present.
Without that slot, fill every required field in the blank mux template.
Keep all tones in their original order. F1/F2 use their captured tone positions.

Both role tones require circle classifiers with radius. Missing classifiers,
linear classifiers and missing radii make the cfg invalid before Run.
Run GE Post under the same readout conditions first.
Classification produces g, e and other. Other stays in the probability denominator.

The two ordinary CZ pulses start from `$coupler.module.cz_f1` and `cz_f2` when
available. Without those slots, enter frequency, signed gain, Nyquist zone and
cosine duration. Generator fields come from captured wiring.
Both pulses need matching frequency, duration and pre-delay on independent
generators. Their post-delays may differ; the sequence waits for both.
Only the two charge lines are driven. The coupler stays DC-biased.

Both Forms list reset, F1/F2 Xπ, dual-line drive and mux readout first.
Captured roles, the `scale` sweep and acquisition settings follow.
Enter a finite, nonnegative, strictly increasing common gain-scale range in `scale`.
The scan multiplies both original signed gains by that scale.
Frequency, waveform, duration, relative phase and signed gain ratio stay fixed.
Set Reps, Rounds and relaxation time for the acquisition budget.
Rounds accumulate counts at acquisition, without a host round loop.

## Run and inspect

The ordinary sequence is
`optional reset → B11 → simultaneous CZ → Xπ⊗Xπ → terminal mux`.
B11 applies both Xπ pulses to the initial population. Without reset, thermal
population remains in the ensemble.

The live figure shows P00 from accumulated terminal counts.
FIT displays all nine same-shot joint probabilities over g, e and other.
P00 is the g,g count divided by the total of all nine classes.
No per-shot IQ or mean-IQ diagnostic is stored or plotted.

The return is a leakage proxy, not physical gate fidelity.
Thermal population, preparation and readout errors remain in it.
Other denotes classifier rejection; it is not a calibrated higher-level population.
A high return alone does not establish a π conditional phase.

## Fit the first complete return

FIT sends P00 to the fixed-zero-phase model
`offset + contrast cos(2π scale/period)`. The candidate common scale is
the measured first positive return, `period`.

Measure scale=0 and continue through at least one full cycle.
FIT uses the contiguous prefix with all requested shots acquired.
It stops that prefix at the first incomplete scale.
A cancelled suffix after a full first cycle still allows a proposal.
Partial scales remain visible but do not enter the fit.
Use enough points to resolve the oscillation; sparse sampling can alias a faster one.

The figure overlays the fit and labels the candidate or failure reason.
The summary includes offset, contrast, period, period standard error and
`fit_quality`. Numeric failure, zero contrast, a degenerate model,
nonfinite uncertainty or incomplete coverage stops the proposal.

Finite identifiable fits remain eligible with low nonzero contrast.
Inspect the measured curve and uncertainty before accepting a proposal.
Thermal mixtures and large-gain AC Stark shifts can violate the single-frequency,
fixed-phase model.

## Preview and selected Apply

Both versions propose the same four independent rows:

| Target | Proposal |
| --- | --- |
| `$coupler.pulses.cz_f1` | Source F1 pulse with gain multiplied by the fitted scale. |
| `$coupler.module.cz_f1` | Concrete key for that pulse in the Run/Load capture. |
| `$coupler.pulses.cz_f2` | Source F2 pulse with gain multiplied by the same scale. |
| `$coupler.module.cz_f2` | Concrete key for that pulse in the same capture. |

Only the gains change. Preview the destination, generator, signed gain,
frequency, relative phase and waveform. Proposals use the saved run cfg and
require matching captured role identities.

Select the rows you accept, then press Apply Selected.
Pulse rows and slot rows apply independently. Apply each required slot explicitly.
Reset other open forms to capture accepted calibrations.
Alternate amplitude and conditional-phase/frequency manually.
After changing pulse shape, duration, phase, gain, frequency or bias,
recheck CZ and remeasure both local-Z corrections.

## Save, load and partial data

Save stores the source cfg and float64 `sums` with shape `Ns × 3 × 3`.
Class axes are terminal F1 then terminal F2, each g, e, other.
Counts remain unnormalized. Storage does not grow with the shot count.
Unmeasured scales are NaN. Cancellation retains completed rounds.

Native Load restores the saved counts and cfg without changing current Form inputs.
The cfg schema is 5.0. Older cfgs using `scales` do not restore.
Their saved counts and axes still load and can be plotted, but fitting requires cfg.
Analysis never looks up current calibrations. Missing saved cfg prevents the fit.
Old per-shot IQ files without sums are rejected; there is no automatic conversion.
`stop_requested` records a stop request, not data completeness.
Acquisition exceptions propagate unchanged from Run. A failed Run publishes no
new record and leaves the previous successful record unchanged, even after Stop.

See the [heralded guide](guide_heralded.md) for post-selection and pass fraction.

## Notebook

Use the adapter with a complete saved or operator-reviewed cfg.
`point_view` is the bound PointView for the save destination.

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
