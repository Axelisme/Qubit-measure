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

**Live** and **Analyze Fit** show raw, equal-scale I/Q with **Probe off** in blue and **Probe on** in red. Live shows the **Acquired shots** count; Fit overlays fitted centers and the threshold perpendicular to the projection. Each IQ bin mixes blue and red using the sixth powers of the two shot fractions, normalized to sum to one. A 60:40 split gives about 92:8 in color; equal counts give purple. Opacity rises with the log of the total shot count, and empty bins are transparent. These are shot distributions, not means with error bars.

**Diagnostic** shows raw, equal-scale I/Q and the fitted centers at top left. The bottom left overlays projected histograms with the fitted curves; `Fraction per bin` is normalized separately for each preparation. The text between the rows gives the prepared-label and ideal-model assignment scores. On the right, compare the fitted components and fractions for g preparation above and e preparation below. **Fitted SNR** is the projected center separation divided by the fitted common width; ± is the standard error from the same joint fit. `unavailable` means that error cannot be estimated, and `not identifiable` means a center failed the relative-error check. Diagnostic IQ maps blue to g preparation and red to e preparation using **Initial state**. R² measures explained histogram variation; normalized residual RMS measures mismatch relative to the histogram range.

**Post-Analysis** uses the selected source's Primary fit and classifies by nearest center inside its radius, not by the Primary threshold. Its left column shows raw, equal-scale I/Q for g preparation in blue above and e preparation in red below. Opacity follows the log of the shot count in each bin; color does not indicate the classified label. Both panels mark the fitted centers, radius boundaries and the boundary midway between the centers where the disks overlap. Read the observed g/e/Other fractions above each panel before comparing the matrices: **Initial Populations** estimates the mixture for each preparation, and **Classifier confusion** estimates measured outcomes by state after removing that mixture. The condition number measures how sensitive the correction is to input errors. Other does not identify a leakage state, and the matrix is not a joint two-qubit assignment calibration.

Primary is preselected only if all four joint-fit checks pass for every included tone: R² ≥ 0.9, normalized residual RMS ≤ 0.1, relative width error ≤ 0.5, and both g/e relative center errors ≤ 0.5; any failed, missing, nonfinite or invalid required metric leaves it unselected.
Post also requires a finite condition number ≤ 10 to be preselected.

Writeback proposes complete replacements of the captured library readout: Primary adds fitted g/e centers to each successful tone's classifier; Post adds centers, radius and confusion matrix. Failed tones keep their previous classifiers. If applying separately, applying Primary after Post removes radius and matrix. Changing readout settings does not invalidate stored classifiers automatically; run GE and Post again before trusting them. For shared controls see [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md).
