**Readers:** operator, agent. Configure two-generator STIRAP scans and analyze acquired g/e/f counts with saved readout calibration.

# STIRAP scans

This experiment starts from relaxed g and drives a Lambda sequence from g to f.
Stokes drives the 2–3 transition. Pump drives the 0–3 transition.
The two Gaussian pulses use different bound generators.
Readout starts after both pulses finish.

Run requires a centroid3 readout classifier before acquisition and saves g/e/f counts.
Analyze normalizes counts and uses only the training confusion from that run's saved readout configuration.
Preparation-label fractions are not gate fidelity.

## Before opening the tab

1. Establish readout timing and frequency with lookback and onetone spectroscopy.
2. At the intended bias, use `twotone/freq` to find 0–1 and 0–3 from ground. Narrow each scan before accepting its frequency.
3. Calibrate a pi pulse on 0–3 with Rabi. In `twotone/freq`, enable `init_pulse` and enter this pi03 preparation. Then scan the probe to find 3–2 from the prepared state.
4. Record the measured 0–3 and 3–2 carriers in MHz. Enter them as Pump carrier and Stokes carrier. Neither field has a model-derived default.
5. Confirm two separately bound charge-drive generators and their allowed gains. The displayed generator numbers are editable starting values, not proof of wiring. In the single-fluxonium simulator, acquire explicitly requires Pump and Stokes to bind declared charge lines; a flux, readout or unbound generator is rejected before evolution. On real hardware, the operator must verify both physical channel roles. Hardware acquisition does not carry or execute this simulator-only check.
6. Prepare g/e/f readout references with `singleshot/qutrit_population`. Analyze the references and Apply the proposed centroid3 classifier to the captured readout library module. Confirm the target and its bound resonator slot before Apply. Reset or reopen STIRAP to capture that module, then Run. Circle rejection is not an f measurement.

Open `New Tab → twotone → stirap`.
A new tab stays Invalid until required carriers, gains, scan controls and readout inputs are filled.
Open or Reset captures the readout source. Changing the library later does not replace the captured input.
Keep enough relax delay to start each shot near g.

## Pulse timing and width

Gaussian sigma is in us. Each pulse lasts six sigma.
The pulses have equal width and no extra pre-delay or post-delay.
A positive Stokes lead delays Pump. A negative lead delays Stokes.
The branches start together, apply the corresponding relative delay, and wait for the later pulse to finish before readout.

Choose a lead smaller than the pulse duration to retain overlap.
For a delay scan, cover both signs and include zero.
A transfer plateau in a later population analysis must be compared against the reversed pulse order, not inferred from timing alone.

## Scan modes

Select delay, amplitude or detuning in the Scan mode row.
Each sweep supports Range or Center/Span coordinates, with point count or step sampling.
A fresh sweep starts with Range and point count. Its coordinates and count remain required.

| Mode | Controls | Horizontal axis | Vertical axis |
| --- | --- | --- | --- |
| delay | Stokes lead start, stop and points | Requested Stokes lead in us | Raw g/e/f fractions |
| amplitude | Common scale start, stop and points; fixed Stokes lead | Dimensionless common scale | Raw g/e/f fractions |
| detuning | Pump and Stokes detuning ranges in MHz; fixed Stokes lead | Pump detuning in MHz | Stokes detuning in MHz |

Amplitude mode multiplies the independently entered Pump gain and Stokes gain by the same scale.
Gaussian sigma stays fixed. All scaled gains must remain within [-1, 1].
To change the Pump/Stokes ratio, edit the two base gains and repeat the scan.

Detuning mode adds each offset to its entered carrier.
The heatmap color is the raw f fraction (f count divided by total count).
It is a Cartesian Pump × Stokes grid, not a paired one-dimensional sweep.
After changing mode, fill its newly shown scan endpoints and point counts.

## MCP recipe

Use `recipe_guide(adapter="twotone/stirap")` to read this guide.
The single `stirap` tool runs the same adapter through Run, Analyze and Save.
Choose `scan_mode` explicitly; the recipe never chooses a mode or tunes controls.

| Recipe input | delay | amplitude | detuning |
| --- | --- | --- | --- |
| `start`, `stop`, `n` | Stokes lead endpoints in us and point count | Common gain-scale endpoints and point count | Pump detuning endpoints in MHz and point count |
| `stokes_start_mhz`, `stokes_stop_mhz`, `stokes_n` | Unused | Unused | Stokes detuning endpoints in MHz and point count |

Delay checks pulse ordering. Amplitude changes both gains at fixed sigma.
Detuning scans a Cartesian Pump × Stokes grid, with positive offsets added to
operator-entered carriers. Each `n` maps to the corresponding saved cfg `expts`.
Use `pump_freq_mhz` and `stokes_freq_mhz` for measured spectroscopy values.
`width_us` is Gaussian sigma; `stokes_lead_us` sets the fixed lead outside delay
mode. Positive lead means Stokes first. Gains use signed native units.
`readout_ref` selects the acquired centroid3 readout library module.

Omitted inputs retain Form values; explicit null goes unchanged to the owner.
Python KEEP marks omission and is not a JSON value. Ordered `overrides` apply
after named inputs and before validation. Missing or invalid Form inputs stop
before Run and retain owner paths and reasons. No model spectrum supplies a
missing frequency. There is no automatic reference acquisition or live-library
calibration lookup during analysis.

The report gives the saved confusion matrix condition number and minimum
singular value. Full raw/corrected g/e/f arrays remain in the analysis summary
and saved artifacts. There is no writeback proposal or question. Analysis
failure still saves acquired g/e/f counts; inspect each Save output receipt rather
than treating recipe completion as proof that every format was saved.
Restart the MCP server after changing recipe declarations; GUI Reload does not
reload that separate process. No separate diagnostic or qutrit-reference tool
is registered.

## Run and saved results

Run classifies the requested repetitions at every scan point into g/e/f counts.
The shots field sets repetitions per point in one round, not a saved shot axis.
The measurement plot shows raw g/e/f fractions, or raw f fraction in detuning mode.
Saved counts have shape (Ny, Nx, 3): Stokes-detuning row, scan-x column and g/e/f.
Delay and amplitude modes have one row.
The configuration records which quantity scan x represents.
Requested coordinates can differ from hardware-quantized timing, gain and frequency.
Unfinished points retain NaN instead of a fabricated result.

A duplicate Pump/Stokes generator is invalid.
Unbound generators fail rather than rerouting the pulse.
Missing readout timing, invalid scan coordinates or out-of-range scaled gains must be corrected before running.
After Run or Load, open Analysis and click Analyze.
Delay and amplitude show separate raw and corrected g/e/f curves.
Detuning shows six heatmaps, with raw fractions above and corrected fractions below.
The title reports the saved training confusion's condition number and minimum singular value.
Analysis output also includes the full training confusion matrix.
Correction keeps negative and above-one estimates. It does not clip or renormalize them.

Run rejects missing or circle/linear classifiers before acquisition.
Analysis rejects missing saved cfg, incomplete/nonfinite or empty counts, and singular training confusion.
Changing the current Form or live library after Run does not change that run's analysis.
Recalibrate, Reset to capture the new readout, and Run again to use new calibration.
This consumer does not train a classifier or offer calibration writeback.
Save All preserves native g/e/f counts, its cfg, analysis summary and figures.
Load analyzes the saved cfg, not the current Form.
Old raw-shot files are rejected with an error naming the missing counts variable.

## Interpretation limits

Measured fractions can change because of transfer, leakage, readout overlap or preparation error.
A three-center classifier reports preparation-label fractions, not certified energy-state populations.
Confusion correction does not remove errors in reference-state preparation.
Simulation results describe the configured model, receiver and noise.
They are not hardware measurements and do not establish gate fidelity.
