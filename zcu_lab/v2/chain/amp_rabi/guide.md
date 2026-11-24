**Readers:** operator, agent. Measure the joint Rabi response to drive gain and compare it with first-read IQ.

# K-cycle readout-chain amplitude Rabi

For opening, refreshing and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

Each rep is one 16-read cycle: `wait K → (gain-swept Rabi pulse → readout → same readout) × 8`. There is no reset inside the cycle.

## Set the sweep

Bind the qubit and its resonator. Select a calibrated single-tone pulse Readout; mux is not supported. Check frequency, drive gain and the readout trigger and integration window before Run. Fill any required blanks. The Rabi Pulse starts from x180 or a fresh qubit probe; no π calibration is required.
- Gain starts at −0.3 to `min(1.0, 1.2 × x180 gain)` when x180 is available. Check that your sweep includes the first oscillation with enough points to locate its extremum.
- **Relax delay** starts at 5 × the captured T1, or 150 µs if T1 is missing. Use at least 5 × T1 for a near-thermal start, including the first cycle.
- Readout `post_delay` runs from pulse end, not ADC close. Check that windows on the same ADC do not overlap and that the next drive has enough ring-down time. Start at `5/(2π·κ)` µs for linewidth κ in MHz; compare with a longer wait before interpreting changes as qubit dynamics. The fallback readout template leaves 0.015 µs of arriving drive on each side of the ADC window.
- Reps counts cycles per point per round. Use at least 66, so a round has at least 65 within-point noise degrees of freedom even for one sweep point. Rounds counts passes through the sweep.

## Read the plots

Run shows `all 65 numbers` and `first read` over completed rounds, both scaled in point-mean standard errors. Both curves subtract their across-gain median. The all-feature direction uses within-point noise and follows the sign of the first-read principal-axis IQ curve. Neither live curve shows shot-SEM bars. Stop keeps completed rounds; stopping before any round finishes fails Run.

Analyze fits all 65 IQ and correlation statistics jointly across the sweep once. The fitter scores trial periods against all 65 features before choosing a starting value. The estimate from the separate first-read cosine is one trial, not the joint estimate.
The joint model interpolates between two population endpoints per initial state. In gain x it uses sin²(πfx) for g and the third state and cos²(πfx) for e. The third state can oscillate if its endpoints differ. Frequency f is in cycles per gain unit and π gain is `1/(2f)`. The dashed joint marker shows π gain inside the scan.

The first-read `fit` shows projected IQ. Error bars are shot SEM: the projected single-cycle standard deviation divided by √(retained cycles at that point), not fit error or round-to-round scatter. If the saved moments cannot supply it, the legend says so and the points have no bars.
With a first-read estimate, points are not connected. Dashed lines and bands show first-read π and π/2 gains and their parameter standard errors. R² describes only this fit.
Without an estimate, the plot says "First-read fit unavailable" and connects the points, keeping any available bars. Joint Analyze still scores trial periods and uses the best trial for its one joint fit. Successful joint Analyze then reports `first_read_pi_amp: null`. A joint fit error still fails Analyze.

The joint figure comes first. Its upper panel overlays measured points and the model along the live all-feature direction. Each bar shows ±1 pooled shot SEM of the projected mean. The estimate pools within-point shot covariance across the sweep and propagates it to the mean using each point's retained cycle count. It treats the projection direction and background as fixed. All bars have the same length in these SEM units. They are not separate noise estimates at each point or fit errors.

The moments pool retained shots across rounds and include projection noise. Changes in the round means contribute to the pooled scatter; the bars neither estimate nor remove drift.

The lower panel shows measured minus model and shades ±1 of the same pooled SEM around zero. Look for ranges where the residuals keep the same sign. If the saved moments cannot support the projection, Analyze still reports the joint fit values and diagnostics and keeps the first-read figure. It draws no joint projection, bars, or residual band.

`J` sums squared residuals whitened by the model covariance across all 65 features. `df` is 65 × the point count minus the joint parameter count. Read `J/df` against the approximate χ² band `1 ± 2√(2/df)` and the fraction of whitened components with `|r| > 3`. These are not the residuals in the lower panel. Neither diagnostic sets a fit acceptance threshold.

The model assumes independent thermalized cycles, Markov transitions, and three shared IQ centers with one isotropic noise width. It does not resolve multiple jumps during readout, cavity transients or the identity of a leakage level. The quoted Wald standard errors measure local parameter uncertainty, not shot SEM or systematic bias. The joint header and summary give π gain and its Wald error; the summary also gives frequency and its Wald error. `R = 1/(2π × gain span × frequency standard error)` measures frequency precision over the sweep, with no cutoff.

## Use the result

Compare the joint π gain with the first-read candidate and check for structure in the residuals. This experiment proposes no writeback; use [`twotone/rabi/amp_rabi`](../../twotone/rabi/amp_rabi/guide.md) to calibrate x180.

`chain_amp_rabi` rejects `null` for `readout_ref` and `drive_ref` at the required Form selector, before Run. For `frequency_mhz`, `pulse_length_us`, `gain_range`, `points`, `reps` and `rounds`, omission keeps the Form value and `null` clears the required value. The readiness check stops before acquisition if it is still blank after overrides. Explicit `gain_range: null` clears both endpoints. Supplying frequency or duration, including `null`, switches the selected Drive pulse to Inline; duration also makes its waveform Inline. Set K with an override on `relax_delay`.
