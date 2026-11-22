**Readers:** operator, agent. Sweep a single two-photon carrier and estimate its g/e/f preparation-label fractions to tune g-to-f transfer.

## Sequence and key settings

One **Raman drive** pulse → calibrated **Readout** → **Relax delay**. There is no reset or same-run reference. This is one carrier, not the two overlapping drives of STIRAP.

- Measure f₀₃ from ground, then f₃₂ after calibrated π₀₃ preparation. Enter **Carrier center** = `(f₀₃ − f₃₂)/2` in MHz; it starts blank. Actual drive RF = carrier center + **Two-photon detuning**/2. Use the measured values, not model predictions.
- **Scan mode** `frequency` holds **Drive gain › Start**; `gain` holds **Two-photon detuning › Start**; `frequency_gain` sweeps both. Gain is a signed DAC full-scale fraction in [−1, 1], not a rotation angle or a common pulse area across waveforms.
- **Raman drive › Waveform** starts at arbitrary and needs an asset; select another available shape to set duration directly. Check generator, gain and timing. Set **Shots per point** for the desired precision and allow enough **Relax delay** for ground preparation.
- Calibrate **Readout** with `singleshot/qutrit_population` at the same readout settings. Run needs the acquired centroid3 classifier, and Analyze needs its confusion matrix. A circle/linear classifier cannot identify f. Recalibrate after changing readout settings.

See [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md) for the shared GUI steps.

## Read the plots

Run shows measured g/e/f label fractions against actual detuning (MHz) or gain (a.u.), or three maps when both vary. The title counts acquired points and retained shots. Stop keeps completed rows; unfinished rows stay blank.

Analyze separates measured and confusion-corrected fractions using the Run's saved calibration, never the current library. For 1D, each bar is one shot SEM: √[p(1−p)/(N−1)] for measured fraction p and N > 1 retained shots. Corrected bars propagate joint label-count covariance, not classifier training uncertainty. The bars are not fit errors, shot spread or scatter between rounds. The 2D maps have no pointwise bars.

A large confusion-matrix condition number or small minimum singular value means correction amplifies count noise. The correction is unclipped; fractions outside [0, 1] are possible. Preparation contamination limits these estimates; they are not pure energy-state populations or gate fidelity. Analysis rejects incomplete, nonfinite or empty counts and singular confusion matrices.

## Next step

Look for consistently high corrected f and low e. If the peak is at +0.8 MHz two-photon detuning, shift the carrier by +0.4 MHz to place it at zero detuning, then check with a finer scan. No writeback is proposed.

Older raw-shot files cannot load. Older 3.x count files can expose counts but cannot restore a typed cfg for Analyze; the current calibration does not fill the gap.

For `raman_gf`, explicit null clears named Form inputs, unlike omission. `waveform_asset` edits the asset of an existing arbitrary waveform: null clears it, and using this input on another waveform family can fail the edit before readiness. For a sampled √sin envelope, use a normalized asset with `√sin(πt/T)` over `0 ≤ t ≤ T`; the sweep does not generate that envelope. Unresolved required fields for the selected mode after overrides fail readiness before Run; a missing centroid3 blocks acquisition. After a successful Run, Analyze failure still leaves the counts available to save.
