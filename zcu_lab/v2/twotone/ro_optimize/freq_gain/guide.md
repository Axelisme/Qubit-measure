**Readers:** operator, agent. Find a readout frequency and gain pair using nominal g/e separation.

# Readout frequency and gain sweep

## Sequence and observable

Optional **Reset** → omit or apply the calibrated **Qubit π pulse** → readout. SNR is the distance between the nominal g/e IQ centroids divided by their pooled single-shot spread along the separation axis. It is not population or measured assignment fidelity.

## Key settings

- Locate the resonator and calibrate the **Qubit π pulse**. Hold readout timing fixed. **Readout frequency** starts at the captured resonator frequency ± half a linewidth. Set it and **Readout gain** to cover the expected peak; Run replaces both coordinates.
- For **Readout pulse and ADC window**, check the timing at the receiver. For readout `post_delay`, start at `5/(2πκ)` µs for linewidth κ in MHz. Compare herald → nominal Xπ at that wait and at a longer one when calibrating this delay.
- **Qubit relaxation wait** starts at 5 × T1 (150 µs if absent). Enter a wait if T1 is invalid.
- **IQ skew penalty weight** of 0 ranks by SNR alone. A positive weight lowers the score for unequal spreads or uncancelled skew along the separation axis.

## Read the figures and choose a setting

Live and Analyze show the same frequency × gain heatmap with measured SNR as color. Analyze labels the selected pair, which may differ from the brightest measured cell if smoothing moves the peak. The map has no per-cell shot SEM.

**Smooth method** and **Smooth strength** affect selection, not the raw measured values. Strength scales the wavelet threshold or sets Gaussian width in sample steps. Missing scores enter smoothing as zeros; check the measured points before using a result from an incomplete sweep.

If the marker is at an edge, extend that axis and repeat. Otherwise tighten around the pair and confirm it. Subtract each old coordinate from the selected one for the corrections. The proposal updates `$qubit.ext.best_ro_freq` and `$qubit.ext.best_ro_gain` and, for a pulse readout with the other calibration coordinate available, the resonator readout module and slot.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
