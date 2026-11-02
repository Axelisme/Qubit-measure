## Measure CPMG coherence

**Readers:** operator, agent. CPMG measures coherence decay for several pi-pulse counts. Each sequence applies pi/2, a train of n refocusing pi pulses, a final pi/2 projection and readout. Calibrate pi, pi/2 and readout before running. The sequence corrects the end intervals for finite pulse widths. Intervals too short to fit the pulses cause Run to fail; increase the minimum time.

**Last updated:** 2026-10-21

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Configure the scan

Pi pulse counts accepts Log or List input. Log uses a maximum count and requested point count. It starts at 1 and retains the maximum count. It rounds geometrically spaced values to integers and removes duplicates. A maximum greater than 1 requires at least two requested points. A maximum of 1 accepts one point and returns `[1]`. Use List for any other single count. The resulting list may have fewer points than requested. List accepts a nonempty JSON array of positive integers, such as `[1, 2, 4, 8]`. Zero, negative counts and fractional counts are invalid.

Minimum time and Maximum time define one shared total-time window in us for every pulse count. Time points sets the samples per row. The initial values are 1 us, 100 us and 101 points. Increase the minimum time when a large pulse count leaves too little spacing between pulses. The same point count applies to every pulse count.

Detune ratio is the number of fringe periods per time index. The default is 0.1. Use Fit fringes for this phase-ramped acquisition. Set Detune ratio to 0 and disable Fit fringes for a decay-only acquisition. Early-stop SNR optionally ends averaging of each row when its signal is resolved.

## Module sources and waiting

The form uses the captured `$qubit.module.x180`, `$qubit.module.x90` and `$resonator.module.readout` slots. Missing slots use pi_pulse, pi2_pulse and readout templates. Reset uses `$qubit.module.reset` and stays disabled when absent. Missing calibration in a template remains Invalid until supplied.

Relax delay starts at 5 times the captured `$qubit.t1`. A missing or null T1 uses 30 us, giving a 150 us wait. An invalid stored T1 leaves the derived input empty. Open and Reset capture these sources. After alias or calibration edits, use Reset to refresh the form.

## Run and analyze

Run displays normalized IQ versus pulse count and time. Save the raw data, then Analyze to fit a decay for each count. The fit figure compares accepted T_CPMG values with pulse count on log axes. Optional Ramsey T2 and T1 references add horizontal T2R and 2*T1 lines in us; the fit uses the measured decay independently of these references.

The heatmap and T_CPMG points remain visible. Incomplete or rejected rows have no T2 point. Analysis fails only when every row fails the per-row fit checks. Extend the time window if the decay is unresolved. Calibration stays unchanged.

The summary fits T2(n) = T2_1 * n^gamma by weighted linear regression of log T2 against log n. Usable rows pass the per-row checks and have finite positive n, T2 and T2 uncertainty. Weights are T2/t2err, so the log-space standard error is t2err/T2. The fit treats these as absolute uncertainties and uses unscaled covariance, without reduced-chi-square scaling. T2_1's standard error is T2_1 times the intercept standard error. The exponent error is the slope standard error.

The summary reports t2_1_us, t2_1_err_us, gamma, gamma_err and fit_rows. With at least three usable rows, the log-log T2 plot overlays the power law and its title reports gamma. With fewer than three usable rows, including zero, all four fit values are null and fit_rows records the usable count. Analysis still succeeds if any per-row fit passed. Zero-error rows remain plotted but do not enter the summary fit. There is no unweighted fallback or artificial uncertainty floor. Per-count T2 values stay in the figure, not the summary.

The cpmg recipe follows the same Run, Analyze and Save steps and reports the same five summary fields. An analysis failure reports null fit values, fit_rows=0 and the failure reason. Use max_n with points for Log input, or times for an explicit list. min_length_us, max_length_us and length_points set the shared time window. Reusing a tab resets it before applying overrides.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Obtain calibrated pi and pi/2 pulses. Set positive refocusing counts and the shared evolution-time window.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, Reset to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, Reset and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `cpmg` after inspecting the same Form errors.
   max_n/points produces the count grid, or supply times. length_points sets Time points and delay.expts. min_length_us/max_length_us set the time window. fit_fringe changes only analysis.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
