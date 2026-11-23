**Readers:** operator, agent. Measure relaxation after a π pulse and compare the chain estimate with first-read IQ.

# K-cycle readout-chain T1

For opening, refreshing and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

Each rep is one 16-read cycle: `wait K → (π pulse → swept wait → readout → same readout) × 8`. There is no reset inside the cycle.

## Set the sweep

Bind the qubit and its resonator. Select a calibrated single-tone pulse Readout; mux is not supported. Check frequency, drive gain and the readout trigger and integration window before Run. Fill any required blanks. Use a calibrated x180.
- Wait after π pulse starts at 0 and stops at 5 × the captured T1, or 150 µs when T1 is missing. Keep points near zero and extend the window to about 3 × tentative T1 if only the start of the decay is visible.
- **Cycle thermalisation wait K** starts at 5 × the captured T1, or 150 µs if T1 is missing. Use at least 5 × T1 for a near-thermal start, including the first cycle.
- Readout `post_delay` runs from pulse end, not ADC close. Check that windows on the same ADC do not overlap and that the next drive has enough ring-down time. Start at `5/(2π·κ)` µs for linewidth κ in MHz; compare with a longer wait before interpreting changes as qubit dynamics. The fallback readout template leaves 0.015 µs of arriving drive on each side of the ADC window.
- Reps counts cycles per point per round. Use at least 66, so a round has at least 65 within-point noise degrees of freedom even for one sweep point. Rounds counts passes through the sweep.

## Read the plots

Run shows first-read IQ on its dominant sweep axis in a.u. (I if too few points define an axis). The lower panel uses all 65 features, centered by their across-sweep median and scaled by pooled noise. It has no shot-SEM bars. The header counts cycles per point after each completed round. Stop keeps completed rounds. Stopping before any round finishes fails Run.
The first-read `fit` figure and upper live panel use shot SEM: the projected single-cycle standard deviation divided by √(retained cycles at that point). Each cycle contributes its first read once. These bars are not fit errors or round-to-round scatter.

Analyze fits all 65 IQ and correlation statistics jointly across the sweep once. The separate first-read single-exponential fit supplies a start, not the reported T1.
When available, the seed curve tracks the first-read points and its dashed marker shows the seed decay time within the scan. In the first-read figure, R² below zero means a worse fit than the mean. Normalized residual RMS is residual RMS divided by the signal range. Neither sets a selection threshold.
Without a seed, the first-read figure says "First-read fit unavailable", connects the points and keeps any available shot SEM. Analyze starts the joint fit from the saved wait span and still fits once. A joint fit error fails Analyze.

The joint figure comes first. Its upper panel overlays measured points and the model along the live all-feature direction. Each bar shows ±1 pooled shot SEM of the projected mean. The estimate pools within-point shot covariance across the sweep and propagates it to the mean using each point's retained cycle count. It treats the projection direction and background as fixed. All bars have the same length in these SEM units. They are not separate noise estimates at each point or fit errors.

The moments pool retained shots across rounds and include projection noise. Changes in the round means contribute to the pooled scatter; the bars neither estimate nor remove drift.

The lower panel shows measured minus model and shades ±1 of the same pooled SEM around zero. Look for ranges where the residuals keep the same sign. If the saved moments cannot support the projection, Analyze still reports the joint fit values and diagnostics and keeps the first-read figure. It draws no joint projection, bars, or residual band.

`J` sums squared residuals whitened by the model covariance across all 65 features. `df` is 65 × the point count minus the joint parameter count. Read `J/df` against the approximate χ² band `1 ± 2√(2/df)` and the fraction of whitened components with `|r| > 3`. These are not the residuals in the lower panel. Neither diagnostic sets a fit acceptance threshold.

The three-state rate model relaxes toward thermal populations. T1 is the eigen time with the larger amplitude of population change, not necessarily the longer one. The secondary time is the other eigen time. Both times are in µs.
The model assumes independent thermalized cycles, Markov transitions, and three shared IQ centers with one isotropic noise width. It does not resolve multiple jumps during readout, cavity transients or the identity of a leakage level. The quoted Wald standard errors measure local parameter uncertainty, not shot SEM or systematic bias.

## Use the result

Compare the two decay modes across repeats. When their amplitudes are close their order can switch, and the fit errors do not include that choice. This experiment proposes no writeback. Use a dedicated [`twotone/t1`](../../twotone/time_domain/t1/guide.md) run to update the qubit calibration.

`chain_t1` rejects `null` for `readout_ref` and `pi_ref` at the required Form selector, before Run. For `max_delay_us`, `points`, `reps` and `rounds`, omission keeps the Form value and `null` clears the required value. The readiness check stops before acquisition if it is still blank after overrides. Set K with an override on `relax_delay`.
