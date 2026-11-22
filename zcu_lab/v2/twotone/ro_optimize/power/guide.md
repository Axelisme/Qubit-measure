**Readers:** operator, agent. Compare g/e separation and state disturbance from paired readouts at different gains.

# Readout gain and QND

## Sequence

Optional **Reset** → preparation I/π → R1 off/on → wait → middle I/π → R2. The three binary choices give eight branches at every gain. I uses the same π pulse at zero gain; R1 off uses zero gain but still triggers the ADC. The R1-to-R2 interval is matched across branches. R1 on and R2 both use the swept gain.

## Key settings

- Calibrate **Qubit π pulse** for both rotations. Set the **Readout gain** range. Check frequency and timing in **Readout pulse and ADC window**; keep both ADC windows covered and disjoint.
- **Wait after R1** is extra time after the generator pulse and its post-delay, not after ADC closure. It starts at 0.01 × T1; **Qubit relaxation wait** after R2 starts at 5 × T1. Without T1, the waits start at 0.3 µs and 150 µs; enter them if T1 is invalid.
- Start readout `post_delay` at `5/(2πκ)` µs for linewidth κ in MHz. Compare herald → nominal Xπ at that wait and at a longer one when calibrating it.

## Read the figures and choose a gain

1. **First-read contrast** is |IQπ−IQI| at R1 on and middle I, against gain (a.u.). Live reports accumulated shots per branch and gain. Analyze keeps the points and marks a manual selection with a red dashed line.
2. **R1-on calibration** compares modeled g/e SNR, Gaussian separation Fsep and thermal excitation. Fsep is not measured assignment fidelity, and the population is not a classified shot fraction.
3. **State disturbance** compares R1 on (circles) with off (squares), using the same R1-on calibration. Retention S and QND Q describe state memory between reads; Q includes relaxation between R1 and R2. The up/down flips and middle π miss are model estimates. Compare mean-based R1-on retention with covariance retention; a mismatch can flag model failure but does not identify leakage.
4. **Separation versus QND** plots one point per gain, colored by gain. The red frontier and starred table rows are the points for which no other measured gain improves one metric without worsening the other. Use the table for the exact gain.

Estimates are not clipped; missing shots or degenerate moments leave gaps. There are no pointwise uncertainty bars, so small advantages need a repeat. Compare adjacent gains before choosing; if the frontier choice is at a scan edge, extend that edge. The gain correction is the table value minus the old gain.

Leave **Selected gain (a.u.)** blank for no proposal. Reanalyze with a finite value in [−1, 1] to propose that exact value, without snapping. The proposal updates `$qubit.ext.best_ro_gain` and, if the Run cfg and other calibration coordinates are available, the resonator DPM readout module and slot.

Old SNR-only files cannot load as QND runs. See the [power storage history](../../../../../docs/storage-format-history/2026-11-01-readout-power-qnd.md).

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
