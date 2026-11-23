**Readers:** operator, agent. Measure Rabi oscillations versus drive duration and compare the joint and first-read π estimates.

# K-cycle readout-chain length Rabi

For opening, refreshing and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

Each rep is one 16-read cycle: `wait K → (length-swept Drive pulse → readout → same readout) × 8`. There is no reset inside the cycle.

## Set the sweep

Bind the qubit and its resonator. Select a calibrated single-tone pulse Readout; mux is not supported. Check frequency, drive gain and the readout trigger and integration window before Run. Fill any required blanks.
- Select a const or flat_top Drive pulse. The starting const probe uses uncalibrated gain 1.0, not x180; confirm the gain before Run.
- Drive duration starts at 0.03 µs and stops at 4 × x180 length (1.0 µs without x180). Extend the window if it misses the first extremum.
- **Cycle thermalisation wait K** starts at 5 × the captured T1, or 150 µs if T1 is missing. Use at least 5 × T1 for a near-thermal start, including the first cycle.
- Readout `post_delay` runs from pulse end, not ADC close. Check that windows on the same ADC do not overlap and that the next drive has enough ring-down time. Start at `5/(2π·κ)` µs for linewidth κ in MHz; compare with a longer wait before interpreting changes as qubit dynamics. The fallback readout template leaves 0.015 µs of arriving drive on each side of the ADC window.
- Reps counts cycles per point per round. Use at least 66, so a round has at least 65 within-point noise degrees of freedom even for one sweep point. Rounds counts passes through the sweep.

## Read the plots

Run shows first-read IQ on its dominant sweep axis in a.u. (I if too few points define an axis). The lower panel uses all 65 features, centered by their across-sweep median and scaled by pooled noise. It has no shot-SEM bars. The header counts cycles per point after each completed round. Stop keeps completed rounds. Stopping before any round finishes fails Run.
The first-read `fit` figure and upper live panel use shot SEM: the projected single-cycle standard deviation divided by √(retained cycles at that point). Each cycle contributes its first read once. These bars are not fit errors or round-to-round scatter.

Analyze fits all 65 IQ and correlation statistics jointly across the sweep once. The separate first-read damped cosine seeds the joint fit via its π duration.
With a seed, the first-read figure marks π and π/2 durations inside the scan. In the first-read figure, R² below zero means a worse fit than the mean. Normalized residual RMS is residual RMS divided by the signal range. Neither sets a selection threshold.
If the seed is unavailable, the first-read figure names the failure, connects the points and keeps any available shot SEM. Joint Analyze uses the saved duration axis for its start and still fits once. A successful joint fit then reports `first_read_pi_len: null`; a joint fit error fails Analyze.

The joint figure comes first. Its upper panel overlays measured points and the model along the live all-feature direction; its lower panel plots measured minus model against the sweep. The scale is the pooled standard error of a point mean, not population or per-point shot SEM. Look for ranges where the residuals keep the same sign.
If the projection is unavailable, Analyze still reports the joint values and diagnostics without these curves and keeps the first-read figure.

`J` sums squared residuals whitened by the model covariance across all 65 features. `df` is 65 × the point count minus the joint parameter count. Read `J/df` against the approximate χ² band `1 ± 2√(2/df)` and the fraction of whitened components with `|r| > 3`. These are not the residuals in the lower panel. Neither diagnostic sets a fit acceptance threshold.

The joint response for a constant pulse is nondecaying with a sin²(πf × duration) drive weight. The π duration is `1/(2f)`, marked in the joint plot when inside the scan. Flat-top ramps contribute rotation that this zero-phase fit does not account for. `R = 1/(2π × sweep span × frequency standard error)` measures frequency precision over the sweep. Unavailable errors are marked in the figure and returned as null in the summary.
The model assumes independent thermalized cycles, Markov transitions, and three shared IQ centers with one isotropic noise width. It does not resolve multiple jumps during readout, cavity transients or the identity of a leakage level. The quoted Wald standard errors measure local parameter uncertainty, not shot SEM or systematic bias.

## Use the result

If the estimates agree within the scan and the fit follows the points, center a dedicated [`twotone/rabi/len_rabi`](../../twotone/rabi/len_rabi/guide.md) calibration on the candidate at the same gain. Increase cycles if the shot SEM hides the oscillation. This chain experiment proposes no writeback.

`chain_len_rabi` rejects `null` for `readout_ref` and `drive_ref` at the required Form selector, before Run. For `frequency_mhz`, `gain`, `max_length_us`, `points`, `reps` and `rounds`, omission keeps the Form value and `null` clears the required value. The readiness check stops before acquisition if it is still blank after overrides. Supplying frequency or gain, including `null`, switches the selected Drive pulse to Inline. Set K with an override on `relax_delay`.
