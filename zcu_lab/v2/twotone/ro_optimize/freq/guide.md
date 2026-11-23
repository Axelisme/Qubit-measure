**Readers:** operator, agent. Find the readout frequency that best separates the nominal g and e preparations.

# Readout frequency sweep

## Sequence and observable

Optional **Reset** → omit or apply the calibrated **Qubit π pulse** → readout. SNR is the distance between the nominal g/e IQ centroids divided by their pooled single-shot spread along the separation axis. It is not population or measured assignment fidelity.

## Key settings

- Calibrate the **Qubit π pulse** and hold readout gain and timing fixed. **Readout frequency** starts at the captured resonator frequency ± one linewidth. Enter RF in MHz with a mixer, DDS frequency without one. The sweep replaces the readout frequency.
- For **Readout pulse and ADC window**, check the timing at the receiver. For readout `post_delay`, start at `5/(2πκ)` µs for linewidth κ in MHz. Compare herald → nominal Xπ at that wait and at a longer one when calibrating this delay.
- **Qubit relaxation wait** starts at 5 × T1 (150 µs if absent). Enter a wait if T1 is invalid.
- **IQ skew penalty weight** of 0 ranks by SNR alone. A positive weight lowers the score for unequal spreads or uncancelled skew along the separation axis.

## Read the figures and choose a setting

Live and Analyze plot SNR versus readout frequency (MHz). Analyze keeps the measured points, overlays the selection SNR if different and marks its measured-grid maximum with a dashed line. No shot SEM or pointwise SNR uncertainty is shown.

**Smooth method** and **Smooth strength** affect selection, not the raw measured values. Strength scales the wavelet threshold or sets Gaussian width in sample steps. Missing scores enter smoothing as zeros; check the measured points before using a result from an incomplete sweep.

Extend the range if the selection is at an edge. Otherwise narrow around the peak and confirm it. The frequency correction is the selected minus the old frequency. The proposal updates `$qubit.ext.best_ro_freq` and, for a pulse readout with the other calibration coordinates available, the resonator readout module and slot.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
