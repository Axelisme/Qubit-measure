**Readers:** operator, agent. Calibrate per-tone g/e single-shot readout for a mux group.

# Mux single-shot GE

Measure raw IQ clouds with and without a calibrated qubit probe. The fitted centers locate g/e response; the reported prepared-label fidelity includes thermal mixing and pulse errors and is not pure-state readout fidelity.

**Sequence:** common reset → simultaneous tone branches (optional **Preparation pulse** in both acquisitions; π probe only in probe-on) → mux readout after the longest branch. The off and on acquisitions are separate; within each one the shot indices align across active tones.

## Key settings

Bind the `mux` alias to a `group/mux` with matching qubits and resonators in readout order. Check the full **Readout** module, including frequencies and ADCs for inactive tones. If there is no readout slot, the inline starting values use group wiring and resonator calibrations; verify gain and timing before Run.

**Tones** chooses what to edit. **Participants** takes a nonempty list of unique zero-based positions in run order, e.g. `[2, 0]`. It selects both drives and emitted readout tones. The `participants` alias seeds those positions; if absent, unbound, or empty, the form starts with all tones. Editing the selection preserves tone values. Fill required pulse and sweep fields for any newly selected tone, using distinct drive generators separate from readout.

Set **Shots** per preparation; there is one round and no averaging into mean-IQ points. For classifier writeback, use the library **Readout** in the group's readout slot without changing frequency, ADC or other acquisition settings. A mismatch prevents the replacement proposal.

**Initial state** labels probe-off as predominantly g or e, not a guaranteed pure state. **Length ratio** is integration length / T1; leave blank to fit it from the decay tail. Without a resolved tail the ratio can be unavailable without changing the centers, fidelity or Post result. Mux **Post-Analysis** chooses the radius automatically.

In `mux_singleshot_ge`, omission retains settings. Null `participants` leaves a missing selection before Run unless an override fills it. Null `initial_state`, `backend`, `logscale` or `align_t1` is rejected by Primary Analyze after acquisition; null `length_ratio` requests estimation. See the [mux recipe inputs](../../../recipes/mux-guide.md#mux_singleshot_ge).

## Read the plots and use the result

Live shows individual **Probe off** and **Probe on** IQ shots. Check the **Acquired shots** count and whether the clouds separate. Analyze overlays fitted centers and a threshold perpendicular to the projection. R² measures explained histogram variation; normalized residual RMS measures mismatch relative to the histogram range. The points are shots, not means with error bars.

Post-Analysis uses the selected Run/Load's Primary fit. It uses nearest-center regions with a radius, not the Primary threshold, and corrects for preparation mixing using fitted populations. Use the classified clouds to check overlap and the fitted initial-population plot to check preparation mixing. Inspect the 3×3 g/e/other matrix and condition number; `other` is not a specific leakage state, and this matrix is not a joint two-qubit assignment calibration.

Primary is preselected only if all four joint-fit checks pass for every included tone: R² ≥ 0.9, normalized residual RMS ≤ 0.1, relative width error ≤ 0.5, and both g/e relative center errors ≤ 0.5; any failed, missing, nonfinite or invalid required metric leaves it unselected.
Post also requires a finite condition number ≤ 10 to be preselected.

Writeback proposes complete replacements of the captured library readout: Primary adds fitted g/e centers to each successful tone's classifier; Post adds centers, radius and confusion matrix. Failed tones keep their previous classifiers. If applying separately, applying Primary after Post removes radius and matrix. Changing readout settings does not invalidate stored classifiers automatically; run GE and Post again before trusting them. For shared controls see [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md).
