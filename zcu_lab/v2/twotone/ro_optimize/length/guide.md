**Readers:** operator, agent. Choose an ADC window for nominal g/e separation, with an option to account for cycle time.

# ADC integration window sweep

## Sequence and observable

Optional **Reset** → omit or apply the calibrated **Qubit π pulse** → readout. SNR is the distance between the nominal g/e IQ centroids divided by their pooled single-shot spread along the separation axis. It is not population or measured assignment fidelity.

## Key settings

- Calibrate the **Qubit π pulse** and hold readout frequency and gain fixed. Set **ADC window** to cover the SNR rise and leveling off. At every point the RF pulse duration is the largest rounded window plus 0.11 µs. Check that the RF pulse reaches the receiver in time to cover the ADC window.
- For **Readout pulse and ADC window**, check the timing at the receiver. For readout `post_delay`, start at `5/(2πκ)` µs for linewidth κ in MHz. Compare herald → nominal Xπ at that wait and at a longer one when calibrating this delay.
- **Qubit relaxation wait** starts at 5 × T1 (150 µs if absent). Enter a wait if T1 is invalid.
- **IQ skew penalty weight** of 0 ranks by SNR alone. A positive weight lowers the score for unequal spreads or uncancelled skew along the separation axis.

## Read the figures and choose a setting

Both plots show SNR versus ADC window (µs). Live counts scored settings; Analyze overlays the selection SNR if different and marks the chosen window. No shot SEM or pointwise SNR uncertainty is shown. Repeat nearby windows if small score differences would change the choice.

**Smooth method** and **Smooth strength** affect selection, not the raw measured values. Strength scales the wavelet threshold or sets Gaussian width in sample steps. Missing scores enter smoothing as zeros; check the measured points before using a result from an incomplete sweep.

Leave **Duration t0 (µs)** blank or zero to maximize selection SNR. Enter a positive fixed time overhead to maximize SNR / √(ADC window + t0) instead. A smaller positive t0 favors shorter windows more strongly. The title reports the objective and the SNR at the chosen window.

Extend the window range if the choice touches an edge. For the change in integration time subtract the old window from the chosen one. The proposal updates `$qubit.ext.best_ro_length` and, for a pulse readout with the other calibration coordinates available, the resonator readout module and slot.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
