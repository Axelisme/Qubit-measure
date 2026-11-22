**Readers:** operator, agent. Compare three sequences at every Rabi gain. After optional preparation reset, apply the gain-swept Rabi pulse; then either read out, apply the tested reset and read out, or apply the tested reset and a second pulse at the same gain before readout.

**Last updated:** 2026-11-22

## Parameter sources

Seeds 51 gain points from 0.0 to 1.0, with relax delay starting at 5*T1. Run overwrites the Rabi pulse gain. Open and Defaults fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.015 µs for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Defaults capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Rabi pulse uses `$qubit.module.x180` or a fresh pi_pulse template. Tested reset uses `$qubit.module.reset` or a fresh pulse-reset template; saved reset families remain selectable. Readout uses the captured `$resonator.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Acquisition and figures

Reps starts at 1000 per sweep point and branch in each acquisition round; Rounds starts at 10. Check the requested Reps × Rounds in Analyze before comparing runs.

Live and the three fit panels use the principal IQ axis of the before-reset branch (raw I if it is unavailable). Comparing the three mean curves shows how much the tested reset erases the first rotation before the second one. The projected IQ is not a population. Each vertical bar is the per-point shot SEM: sample standard deviation along that same IQ axis divided by the square root of the retained shot count. It measures uncertainty of the mean, not shot scatter or fit-parameter uncertainty. Files saved before shot moments were recorded still load and show points without bars.

In Analyze, points are measured means and solid lines are models. All branches use the before-reset frequency f. The reset-plus-Rabi model adds a 2f term. Read branch amplitudes in a.u., f in cycles/a.u., and the after/before amplitude ratio in a.u. The latter is relative contrast, not fidelity. The separate residual plot shows measured minus fitted IQ; RMS gives the typical residual size (a.u.). A structured residual can reveal oscillations not captured by the model. There is no automatic fit threshold for writeback. Phase estimates for a weak fundamental may be unavailable; an unavailable phase is not a zero offset.

A usable reset has small residual oscillation in the reset-only branch and recovers contrast in the reset-plus-Rabi branch. Compare the tested settings using their before-normalized contrast and the residual panel. To sample the fitted oscillations more densely, use a gain step below 1/(4f) to resolve 2f; extend the sweep toward at least one full period, 1/f, if the trace covers less. Confirm the chosen setting with a population measurement before calling it ground-state preparation.

Layout reference: Ding, Fig. 5-13(b) (printed p. 157) separates measured points and model lines for benchmarking. These traces instead compare reset sequences on average projected IQ versus Rabi gain.

## Typical writeback

Analysis reports results for review; calibration stays unchanged. Compare relative contrast, reset-only residual oscillation and fit residuals. Second-harmonic amplitude can reflect coherence or reset error. Ground-state preparation requires independent validation even when the reset-only trace is flat.

## Recommended

Use at least six finite points per branch, preferably 51 points spanning a full Rabi period with enough density to resolve 2f. Frequency is in cycles/gain, phases in degrees at gain zero; phase is unavailable when the fundamental is unresolved above residual noise. Partial sweeps spanning less than a period may give poorly constrained fits. relax_delay should be long enough for thermal equilibration (notebook: 5 × T1 for bath reset). The upstream reset is optional; keep it disabled until calibrated.
