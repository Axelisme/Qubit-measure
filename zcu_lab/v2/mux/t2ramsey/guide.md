**Readers:** operator, agent. Read mux Ramsey coherence and review frequency corrections.

# Mux Ramsey

Measure free-evolution coherence from mean IQ as the time between π/2 pulses changes. The decay envelope gives T2*; the fitted fringe rate includes the applied phase ramp and the drive-to-transition mismatch.

**Sequence:** common reset → simultaneous π/2 pulses → common delay → simultaneous phase-ramped π/2 pulses → mux readout. The delay starts after the longest first branch; shorter branches can accrue an extra fixed wait and phase.

## Key settings

Bind the `mux` alias to a `group/mux` with matching qubits and resonators in readout order. Check the full **Readout** module, including frequencies and ADCs for inactive tones. If there is no readout slot, the inline starting values use group wiring and resonator calibrations; verify gain and timing before Run.

**Tones** chooses what to edit. **Participants** takes a nonempty list of unique zero-based positions in run order, e.g. `[2, 0]`. It selects both drives and emitted readout tones. The `participants` alias seeds those positions; if absent, unbound, or empty, the form starts with all tones. Editing the selection preserves tone values. Fill required pulse and sweep fields for any newly selected tone, using distinct drive generators separate from readout.

Use calibrated π/2 pulses. Set each tone's **Detune** in MHz: the final pulse phase advances by detune × delay cycles. Choose a shared **Delay** window of about 3 × expected T2* with at least 10 points per fringe, based on the fastest expected rate. A short window shows only part of the decay, so T2* relies on extrapolation; sparse points can hide or alias fringes.

**Fit fringe** uses a damped cosine (at least six finite points); off uses an exponential (at least four). **Fit phase offset (fringe only)** lets the fit find a constant offset; otherwise the phase follows the 0°/180° convention.

In `mux_t2ramsey`, omission retains settings. Null `participants` leaves a missing selection at the pre-Run readiness check unless an override fills it. Null `fit_fringe` or `fit_phase` is rejected by Analyze, after acquisition. See the [mux recipe inputs](../../../recipes/mux-guide.md#mux_t2ramsey).

## Measurement and analysis

Live `measurement_i` and Analyze `fit_i` use the original tone index `i`, not the participant-list row. Both show the signed principal-axis projection of mean IQ (a.u.) against requested delay. If the fit fails, `fit_i` shows the acquired points and the failure reason instead of a fitted curve. With fewer than two finite points, the plot uses the I axis. The projection is not population. There are no pointwise shot SEM bars; the ± on T2 is fit uncertainty.

R² is explained variation and normalized residual RMS is mismatch relative to signal range. A weak fit, unresolved fringe or cross-tone contamination makes a correction unreliable. Sampling uses `1 / (largest delay gap × max(|fitted fringe rate|, |actual detune|))`; adequate means at least 10 points per cycle and a full cycle, not proof against unknown aliases. Changing Participants changes both drives and readout; a single-participant trace does not isolate drive effects.

## Next step

Writeback proposes `t2r`, a T2R sample at captured bias, and, with valid source and fringe inputs, a frequency correction. T2/sample preselection requires positive finite T2*, valid nonnegative relative T2 error ≤ 0.5, R² ≥ 0.9 and normalized residual RMS ≤ 0.1. Frequency also needs adequate sampling and valid nonnegative relative rate error ≤ 0.5; a missing, nonfinite or invalid required value leaves the candidate unselected.

Use `new frequency = π/2 source carrier + actual detune − fitted fringe rate` (MHz) only after verifying sign and phase wrap. Actual detune comes from the applied quantized phase ramp, not the nominal Form value. If the fitted rate exceeds actual detune by 0.002 MHz, the correction is −0.002 MHz; an unavailable detune prevents the frequency proposal. Prefer using this measurement for the T2* trend. Recalibrate pulses after a frequency update; the writeback does not change existing pulse modules. For shared controls see [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md).
