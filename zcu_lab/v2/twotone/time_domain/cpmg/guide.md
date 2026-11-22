## Measure CPMG coherence

**Readers:** operator, agent. CPMG measures coherence decay for several pi-pulse counts. Each sequence applies pi/2, a train of n refocusing pi pulses, a final pi/2 projection and readout. Calibrate pi, pi/2 and readout before running. The sequence corrects the end intervals for finite pulse widths. Intervals too short to fit the pulses cause Run to fail; increase the minimum time.

**Last updated:** 2026-11-22

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Configure the scan

Pi pulse counts accepts Log or List input. Log uses a maximum count and requested point count. It starts at 1 and retains the maximum count. It rounds geometrically spaced values to integers and removes duplicates. A maximum greater than 1 requires at least two requested points. A maximum of 1 accepts one point and returns `[1]`. Use List for any other single count. The resulting list may have fewer points than requested. List accepts a nonempty JSON array of positive integers, such as `[1, 2, 4, 8]`. Zero, negative counts and fractional counts are invalid.

Minimum time and Maximum time define one shared total-time window in µs for every pulse count. When you open the tab or choose Defaults, Minimum time starts at 16 times the sum of the calibrated π/2 and π pulse durations. The factor 16 is the initial maximum number of refocusing pulses. If either calibrated pulse is missing or uses an arbitrary waveform defined by samples, the form leaves Minimum time blank. Enter it before running. Changing Pi pulse counts does not recalculate Minimum time. Maximum time starts at 100 µs, and Time points starts at 101 samples per row. Increase the minimum time when a large pulse count leaves too little spacing between pulses.

For Python callers, `CPMG_Cfg.delay` also accepts a list of `SweepCfg`, one delay sweep per pi-pulse count in `count` order. Every sweep must have the same number of points. This lets different pulse counts use different total-time windows. The GUI continues to provide one shared window.

`Detune ratio (fringes/step)` sets the number of fringe periods per total-time sweep step. The default is 0.1. Use Fit fringes for this phase-ramped acquisition. Set Detune ratio to 0 and disable Fit fringes for a decay-only acquisition. Early-stop SNR optionally ends averaging of each row when its signal is resolved.

## Module sources and waiting

The form uses the captured `$qubit.module.x180`, `$qubit.module.x90` and `$resonator.module.readout` slots. Missing slots use pi_pulse, pi2_pulse and readout templates. Reset uses `$qubit.module.reset` and stays disabled when absent. Missing calibration in a template remains Invalid until supplied.

Relax delay starts at 5 times the captured `$qubit.t1`. A missing or null T1 uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves the derived input empty. Open and Defaults capture these sources. After alias or calibration edits, use Defaults to refresh the form.

## Run and analyze

Run shows the Live heatmap and adds traces for the latest and previous acquisitions as data arrives. Save the raw data, then Analyze to fit a decay for each count. The Analyze figure shows the contrast heatmap on the left and fitted lifetimes on the right. See [Reading the plots](#reading-the-plots) for the axes in each panel. Optional Ramsey T2 and T1 references add horizontal T2R and 2*T1 lines in µs. The fit does not use these references.

Analyze skips incomplete rows and omits lifetimes for rows that fail the fit checks. A fit figure requires at least one row to pass. If no row passes or a fitter raises an error, Analyze fails without producing a new fit figure. A fitter error stops Analyze even if another row passed. Extend the time window if the decay is unresolved. Calibration stays unchanged.

The summary fits T2(n) = T2_1 * n^gamma by weighted linear regression of log T2 against log n. Usable rows pass the per-row checks and have finite positive n, T2 and T2 uncertainty. Weights are T2/t2err, so the log-space standard error is t2err/T2. The fit treats these as absolute uncertainties and uses unscaled covariance, without reduced-chi-square scaling. T2_1's standard error is T2_1 times the intercept standard error. The exponent error is the slope standard error.

The summary reports t2_1_us, t2_1_err_us, gamma, gamma_err and fit_rows. With at least three usable rows, the log-log T2 plot overlays the power law and its title reports gamma. With fewer than three usable rows, including zero, all four fit values are null and fit_rows records the usable count. Zero-error rows remain plotted but do not enter the summary fit. There is no unweighted fallback or artificial uncertainty floor. Per-count T2 values stay in the figure, not the summary.

The cpmg recipe follows the same Run, Analyze and Save steps and reports the same five summary fields. An analysis failure reports null fit values, fit_rows=0 and the failure reason. Use max_n with points for Log input, or times for an explicit list. min_length_us, max_length_us and length_points set the shared time window. Reusing a tab resets it before applying overrides.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Obtain calibrated pi and pi/2 pulses. Set positive refocusing counts and the shared evolution-time window.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, choose Defaults to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, choose Defaults and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `cpmg` after inspecting the same Form errors.
   max_n/points produces the count grid, or supply times. length_points sets Time points and delay.expts. min_length_us/max_length_us set the time window. fit_fringe changes only analysis.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Defaults may replace unsaved inputs.

## Reading the plots

Both heatmaps put each acquisition in an equal-width column in scan order. Column spacing shows acquisition order, not the numerical distance between counts. Each tick labels the actual π pulse count. If a count repeats, its tick also shows its acquisition position, starting from 1. For example, `4 (#3)` means that the third acquisition used four π pulses.

Each column in both heatmaps uses its own total evolution delays in µs. Its color shows IQ scaled to that column's observed contrast range, not a population. Unmeasured cells stay blank.

- **Live right panel:** The latest and previous acquisitions each show normalized IQ against their own delay grid.
- **Analyze right panel:** The fit plots T₂ against the numerical pulse count on log-log axes, not the heatmap's acquisition positions. Repeated counts share the same x position. Error bars show the standard error of each fit, not shot SEM. Incomplete or rejected rows leave no lifetime point.

For implementation, see the [live plotting tools for experiment authors](../../../README.md#即時圖工具).

Rows must have resolved decay within the measured window, fit error below half the lifetime, and fitted contrast above four times the mean absolute smoothed residual. Missing or rejected rows do not contribute to the power law. Widen the total-delay window if a fitted lifetime reaches the grid span; increase Reps or Rounds if the decay is obscured by noise. The Analyze summary reports configured Reps × Rounds.
