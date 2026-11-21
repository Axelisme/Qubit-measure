**Readers:** operator, agent. Compare the separation and state disturbance of paired readouts at different gains.

# Readout gain and QND

## Before running

Select the qubit and its resonator. Use **Defaults** after changing either source. Confirm the starting values before Run:

| Form field | What to enter |
| --- | --- |
| Qubit π pulse | Select the calibrated g-to-e pulse for preparation and the middle pulse. Defaults uses the captured x180 slot or the qubit's π-pulse template. |
| Readout pulse and ADC window | Select the captured resonator readout or its template. Check frequency, waveform and integration window. The template leaves 0.015 µs of pulse on each side of the ADC window. |
| Reset | Select qubit reset before each preparation, or leave blank to omit reset. |
| Readout gain (a.u.) | Set the gain range. The initial 101 points span 0.001 to 0.2. Run uses the sweep gain for R1 on and R2. |
| Wait after R1 (µs) | Enter the extra time between the end of the R1 waveform and post-delay and the start of the middle pulse. Defaults starts at 0.01 × captured T1. |
| Qubit relaxation wait (µs) | Set the wait after R2. Defaults starts at 5 × captured T1. |
| Reps (count) | Set shots per branch and gain per round. Defaults starts at 1000. |
| Rounds (count) | Set repetitions of the full gain scan. Defaults starts at 10. |

If no T1 is available, the two waits start at 0.3 µs and 150 µs, based on a 30 µs starting estimate. Enter waits yourself if the saved T1 is invalid. R1 post-delay starts at the end of the generator pulse, not at the end of integration. Use five photon decay times, `5/(2πκ)` µs for linewidth κ in MHz, as a starting point. Check that the two ADC windows do not overlap and that a longer post-delay does not change the result.

## Sequence and stored data

At every gain, the experiment acquires eight branches:

| Branch | Preparation | R1 | Middle pulse |
| --- | --- | --- | --- |
| 0 | I | off | I |
| 1 | I | off | pi |
| 2 | I | on | I |
| 3 | I | on | pi |
| 4 | pi | off | I |
| 5 | pi | off | pi |
| 6 | pi | on | I |
| 7 | pi | on | pi |

Each branch runs reset if enabled, preparation, R1, wait, middle pulse and R2 in that order. Both preparation I and middle I play the configured qubit pulse at zero gain. Pi uses its configured gain. I and pi therefore occupy the same native slot, including the waveform and configured pre-delay and post-delay. R1 off also plays a zero-gain readout pulse and still triggers the ADC. R2 always plays at the swept gain.

All eight branches have the same reset, preparation, R1, wait, middle and R2 positions. Their R1-to-R2 trigger interval is equal across I/pi and R1-on/off choices. The program emits the zero-gain pulses rather than estimating pulse durations and inserting delays.

## Read the figures

1. **First-read contrast.** Live plots the magnitude of the difference between mean IQ for π and I preparations, with R1 on and middle I. The x axis is readout gain (a.u.). The y axis is a contrast in a.u., not SNR or population. The live heading reports accumulated shots per branch and gain. Analyze keeps the same points and marks a manual selection, if supplied, with a red dashed line.
2. **R1-on calibration.** Compare g/e SNR, Gaussian overlap estimate Fsep and the estimated thermal excited-state population across gains. Fsep is the separation predicted by the two-state Gaussian model, not assignment fidelity measured from labeled shots. The estimated population comes from the model, not classified counts.
3. **State disturbance.** Compare R1-on circles and R1-off squares for retention S, QND Q, up and down flip probabilities, and the middle π miss fraction. These are model estimates. Compare the R1-on mean-based retention with the covariance retention in the last panel. A disagreement can indicate failure of the two-state, isotropic-noise model. It cannot identify leakage.
4. **Separation versus QND.** Each point is one measured gain; the colorbar gives gain in a.u. The red Pareto frontier keeps settings where no other point improves one estimate without worsening the other. Read off the exact gain in the lookup table. An asterisk marks a frontier point.

The QND calculation includes relaxation during the R1-to-R2 interval. Compare R1 on and off before attributing QND loss to the measurement. Analysis uses the R1-on calibration for both because R1 off has no state contrast. It shows estimates without clipping. Missing shots or degenerate moments leave gaps. None of the plotted metrics is a linear mean or a classified shot fraction. The plots do not show shot SEM for these derived estimates.

Choose a frontier point that trades enough separation for the required retention and compare adjacent gains before using it. If your current gain is 0.10 a.u. and the row you choose is 0.14 a.u., enter 0.14 a.u. for an increase of 0.04 a.u. If your choice lies on a scan edge, repeat with a range that extends in that direction before using it.

The Analyze summary reports the source run's Reps × Rounds if its cfg is available. The first-read contrast figure shows the manual selection, if given. Native data retains the gain axis, paired IQ moment sums and shot counts for each branch. It does not retain the raw shots.

## Choosing and saving a gain

Leave `Selected gain (a.u.)` empty to inspect plots without any writeback. Enter a finite gain within `[-1, 1]` and Analyze again to propose that human-selected value. The adapter proposes `$qubit.ext.best_ro_gain`. When the captured best readout frequency and length are also available, it proposes the resonator DPM readout module and its readout slot through the existing writeback path. Missing cfg or calibration coordinates omit the module proposal. Review the proposals before applying them. The chosen value is not snapped to a sweep point.

The GUI experiment key remains `twotone/ro_optimize/power`. Native QND files use tag `twotone/ro_optimize/power_qnd` and cfg schema `1.0`. Load rejects the old SNR-only power tag. Those files lack paired moments and cannot be converted to QND metrics. Keep them for historical analysis with the matching older software.

There is no MCP recipe for this experiment. Use the experiment tab or its RPC entry point.
