**Readers:** operator, agent. Configure two-generator STIRAP scans and analyze acquired IQ with saved readout calibration.

# STIRAP scans

This experiment starts from relaxed g and drives a Lambda sequence from g to f.
Stokes drives the 2–3 transition. Pump drives the 0–3 transition.
The two Gaussian pulses use different bound generators.
Readout starts after both pulses finish.

Run saves raw acquired IQ. Analyze uses only the three-center classifier and training confusion from that run's saved readout configuration.
Do not interpret mean IQ curves or heatmaps as populations or gate fidelity.

## Before opening the tab

1. Establish readout timing and frequency with lookback and onetone spectroscopy.
2. At the intended bias, use `twotone/freq` to find 0–1 and 0–3 from ground. Narrow each scan before accepting its frequency.
3. Calibrate a pi pulse on 0–3 with Rabi. In `twotone/freq`, enable `init_pulse` and enter this pi03 preparation. Then scan the probe to find 3–2 from the prepared state.
4. Record the measured 0–3 and 3–2 carriers in MHz. Enter them as Pump carrier and Stokes carrier. Neither field has a model-derived default.
5. Confirm two separately bound charge-drive generators and their allowed gains. The displayed generator numbers are editable starting values, not proof of wiring.
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
| delay | Stokes lead start, stop and points | Requested Stokes lead in us | Mean I and Q in ADC units |
| amplitude | Common scale start, stop and points; fixed Stokes lead | Dimensionless common scale | Mean I and Q in ADC units |
| detuning | Pump and Stokes detuning ranges in MHz; fixed Stokes lead | Pump detuning in MHz | Stokes detuning in MHz |

Amplitude mode multiplies the independently entered Pump gain and Stokes gain by the same scale.
Gaussian sigma stays fixed. All scaled gains must remain within [-1, 1].
To change the Pump/Stokes ratio, edit the two base gains and repeat the scan.

Detuning mode adds each offset to its entered carrier.
The heatmap color is the magnitude of mean acquired IQ.
It is a Cartesian Pump × Stokes grid, not a paired one-dimensional sweep.
After changing mode, fill its newly shown scan endpoints and point counts.

## Run and saved results

Run acquires the requested raw shots at every scan point.
The measurement plot shows the mean of those shots.
Saved raw IQ has axes Stokes-detuning row, scan-x column and shot.
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

Missing saved cfg, missing classifier, circle/linear calibration, incomplete shots or singular training confusion fail explicitly.
Changing the current Form or live library after Run does not change that run's analysis.
Recalibrate, Reset to capture the new readout, and Run again to use new calibration.
This consumer does not train a classifier or offer calibration writeback.
Save All preserves native raw IQ, its cfg, analysis summary and figures. Load analyzes the saved cfg, not the current Form.

## Interpretation limits

Mean IQ can change because of transfer, leakage, readout overlap or preparation error.
A three-center classifier reports preparation-label fractions, not certified energy-state populations.
Confusion correction does not remove errors in reference-state preparation.
Simulation results describe the configured model, receiver and noise.
They are not hardware measurements and do not establish gate fidelity.
