**Readers:** operator, agent. Measure the resonator center and full linewidth (FWHM) from a narrow one-tone sweep of complex IQ.

# One-tone resonator spectrum

Each shot plays one readout pulse → demodulates at that same swept RF → integrates over the ADC window. Each round covers the full RF grid. There is no qubit drive or state classification. Analyze fits both quadratures: the center locates the resonance, and FWHM gives its frequency width.

## Set up the sweep

- Bind `resonator` and measure flight time with [lookback](../../lookback/guide.md). If the center is unknown, start with [range find](../range_find/guide.md). Check **Gen ch**, **RO ch**, **NQZ** (output Nyquist zone), and **Mixer freq** (blank for a route without a mixer) against that wiring. **Gain** is DAC full-scale fraction, not power. Leave **Classifier** at None.
- **Readout › Phase** rotates the transmitted carrier in degrees; it is not the Homophasal circle phase. **Pre-delay** moves the transmitted pulse but not **Trig offset**. Move the trigger by the same amount if the ADC window should stay at the same place on the pulse.
- **Trig offset** starts at flight time plus `5/(2π·κ)` µs for linewidth κ in MHz, or flight time alone without κ. The pulse starts no shorter than the template and extends to cover the trigger plus **RO length** (template length minus 0.03 µs). If flight time is missing, fill the required trigger and pulse length. Check the ADC window against the arriving pulse.
- **Post-delay** lets the cavity empty. Start at `5/(2π·κ)` µs; to calibrate this wait, compare herald → nominal Xπ with a longer wait. **Relax delay** is the separate wait after each shot.
- **Readout RF frequency** starts at the captured center ± 1.5 linewidths. In `linrange` enter Start/Stop; in `center` enter Center and full Span (endpoints Center ± Span/2). Both use Expts and include the endpoints. The `_step` modes use a Step toward the stop, no larger than the span; a partial final step is dropped. For example, 0.3 MHz over 1 MHz stops 0.9 MHz from the start.
- Use **Sampling mode › Linear** for a first fit. **Homophasal** keeps the effective endpoints and count but spaces samples in circle phase. Use the prior fitted **Resonance frequency**, **Linewidth (FWHM)** and **Circle phase offset** (rad), not the transmitted pulse phase. If the old calibration is unreliable, return to Linear.

For tab setup, refreshing sources, applying and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

## Read the plots

- **Run** plots `|mean(I + iQ)|` (a.u.) versus acquired absolute RF (MHz), with no shot SEM. Widen the scan if either flank is clipped. Increase averaging when noise hides the feature.
- **Analyze** overlays the fit on the magnitude samples and marks its center, which need not coincide with the notch minimum. The fit excludes the first and last samples. FWHM and center are in MHz, without fit standard errors. Choose **Resonance model** `hm` for a hanger notch, `t` for a transmission peak or `auto` to choose from magnitude. The two background controls allow an amplitude slope and quadratic phase curvature; drop a term if the scan cannot distinguish it from the resonance.
- **Electrical delay seed** (µs) selects a starting branch, not a fixed delay. Blank uses the saved delay for the same generator/ADC pair or searches within 100 µs without one. Enter a signed seed (including zero) only with independent evidence from calibration or a prior spectrum; ADC flight time is not that evidence.
- **IQ diagnostics** compares unwrapped phase and the IQ locus against the fit. Check for systematic mismatch even when magnitude agrees. R² compares with the mean predictor (1 exact, 0 as good as the mean, negative worse); normalized residual RMS divides RMS mismatch by the observed range. Both use the interior I and Q, not the magnitude alone. The summary's Reps × Rounds are configured, not completed shots.

## Use the fit

For a refined scan use `center_fit ± 1.5·FWHM_fit` with several samples per linewidth. The center correction is `center_fit − center_previous`. Confirm state separation separately; a resonance fit does not measure it.

Writeback proposes resonator frequency, linewidth, circle phase, and a readout differing only in RF frequency (accept the module and slot rows together to select it). Delay is proposed only for a known generator/ADC route and a trusted seed or nonuniform grid; a uniform grid alone leaves the branch ambiguous. A native Load without compatible parameter capture can still fit IQ, but does not supply a source for writeback.

## Recipe inputs

`onetone_spectrum`: omit both `center_mhz` and `span_mhz` to keep the sweep. Supplying only one, even null, returns `needs_parameters` before Run. Supplying both with either null clears both endpoints; unless `overrides` restores them, Run readiness fails.

Null `readout_ref` is rejected when selecting the required Readout, before Run. Supplying `gain` converts Readout to inline before setting the value. Null `gain`, `points`, `reps` or `rounds` goes to the Run Form; unresolved required fields fail readiness after overrides. Null `model_type`, `fit_bg_amp_slope` or `fit_bg_phase_curvature` is rejected at Analyze, after acquisition. Null `edelay` selects the blank/automatic policy above. The recipe saves completed Run data even if Analyze fails.
