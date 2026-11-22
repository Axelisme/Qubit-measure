# Single-shot GE readout calibration

**Readers:** operator, agent. Calibrate readout from two IQ clouds acquired with the π probe off and on.

## What is measured

Each shot runs **Reset** (optional) → **Init Pulse** (optional) → **Probe Pulse** (off or on) → **Readout** → **Relax delay**. Run takes **Shots** readouts per branch, in one round per branch.

Primary rotates the IQ plane and fits the projected histograms to estimate the g/e centers, common width and preparation mixtures. The rotation angle and threshold define the projected boundary. Fidelity is the equal-weight assignment score for the two *prepared labels* at that boundary, not pure-state readout fidelity. Probe off/on can each contain a mixture, and a threshold can exploit differences within a single cluster.

## Key settings

- Before Run, verify **Probe Pulse** is a calibrated g↔e π pulse and **Readout** has calibrated settings and timing. Bind the intended readout module. If starting from the fresh readout template, trigger timing starts at the stored resonator flight time + 0.015 µs; verify the actual ADC window. Resolve required fields marked Invalid.
- **Shots** samples each cloud. Increase it if the fitted histogram is poorly sampled. **Relax delay** starts at 5 × stored ordinary T1 (150 µs if T1 is absent); invalid T1 leaves it blank. Reset does not shorten this wait automatically.
- **Initial State** names the predominant state after reset/init and before the probe. If it is `excited`, probe off maps to e and probe on to g. **Backend** rotates along the pooled PCA axis (`pca`) or the line between the centers (`center`).
- **Align T1** shares the decay ratio across histogram fits. **Length ratio (a.u.)** is readout duration / T1. Leave it blank to estimate it from the tail between the clouds. With no identifiable tail, Analyze reports `length_ratio: null` and labels the scatter `GE IQ (length ratio: not identifiable)`; the missing ratio does not itself invalidate the centers, fidelity or Post result. A supplied ratio is reported as entered. **Log Scale** does not change the displayed scatter.
- In `singleshot_ge`, omitted experiment inputs keep the form or Primary settings; `use_reset: null` and `init_pulse_ref: null` disable those modules. Null `readout_ref` or `pi_ref` fails at required-module selection. Null `shots` clears the shot count; if no override fills it, the readiness check returns `needs_parameters` before Run. Null `initial_state`, `backend`, `logscale` or `align_t1` fails at Primary option validation, after Run; null `length_ratio` requests estimation. The recipe saves the completed Run even if analysis fails.

## How to read the results

- **Live** shows individual I/Q shots in a.u., grouped by probe off/on. **Analyze** shows the same shots labeled by g/e preparation using **Initial State**. Separate clouds support discrimination; heavy overlap does not. These scatters have no shot SEM bars.
- For the joint histogram fit, R² measures explained variation, normalized residual RMS measures the size of the mismatch, and the relative width and center errors compare fitted standard errors with the magnitudes of the fitted values. A poor fit makes inferred preparation mixtures and the Post correction unreliable, even if an assignment score is available.
- **Post-Analysis** classifies by nearest center and radius, not the Primary threshold. Compare each preparation's classified IQ and fractions with **Initial Populations** (fitted preparation mixture), then **Confusion Matrix** (inferred state → measured outcome after removing that mixture). Other (`|L⟩`) is a rejected class, not an identified leakage level. The condition number measures how sensitive the correction is to input errors.

## Next step

Primary is preselected only if the joint fit has R² ≥ 0.9, normalized residual RMS ≤ 0.1, relative width error ≤ 0.5, and g/e relative center errors each ≤ 0.5; if any check fails or a required metric is missing, nonfinite or marked invalid, it is not preselected.
Post also requires a finite condition number ≤ 10 for preselection. A candidate left unchecked is still selectable. Inspect IQ separation and fit quality before accepting either proposal; if the clouds overlap, check preparation and readout before rerunning.

Writeback replaces the Run's bound readout module classifier: Primary writes g/e centers, and Post adds radius and the 3×3 confusion matrix. Primary applied *after* Post clears the radius and matrix. Post can be accepted on its own; the recipe asks separately for each stage. Writeback requires the measured readout settings to match the captured library module apart from its classifier. Changing readout settings does not invalidate the stored classifier automatically; rerun GE and Post before using it for the new settings.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
