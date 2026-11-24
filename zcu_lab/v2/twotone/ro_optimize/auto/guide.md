**Readers:** operator, agent. Find a measured readout RF, gain and RF pulse duration triple with high nominal g/e separation.

# Readout search

## Sequence and observable

Optional **Reset** → omit or apply the calibrated **Qubit π pulse** → readout. SNR is the distance between the nominal g/e IQ centroids divided by their pooled single-shot spread along the separation axis. It is not population or measured assignment fidelity.

## Key settings

- Calibrate the **Qubit π pulse**. **Readout frequency** bounds start at the resonator frequency ± 0.2 linewidths. Set **Readout gain** and **Readout pulse duration** bounds; the duration changes RF, not the ADC window, so check coverage at the minimum. **Search candidate count** sets the budget. The search uses continuous bounds, not an Expts grid, and measured scores guide subsequent candidates. Supply **Search random seed** to repeat the sequence given the same scores; blank starts a new sequence.
- For **Readout pulse and ADC window**, check the timing at the receiver. For readout `post_delay`, start at `5/(2πκ)` µs for linewidth κ in MHz. Compare herald → nominal Xπ at that wait and at a longer one when calibrating this delay.
- **Qubit relaxation wait** starts at 5 × T1 (150 µs if absent). Enter a wait if T1 is invalid.
- **IQ skew penalty weight** of 0 ranks by SNR alone. A positive weight lowers the score for unequal spreads or uncancelled skew along the separation axis.

## Read the figures and choose a setting

The four panels show SNR by candidate index, readout frequency (MHz), gain (a.u.) and pulse duration (µs). Only the candidate-index panel connects scores; the other panels show coverage by setting. Analyze marks the best measured triple with a red star in each panel. Points sharing one coordinate can differ in the other two; these are not independent sweeps. The nonlinear score has no plotted pointwise uncertainty.

Compare the winner with other high scores. Extend a bound if good candidates cluster there and it is safe to extend; use more candidates if coverage is sparse. Confirm the triple in a new measurement. The proposal updates `$qubit.ext.best_ro_freq`, `$qubit.ext.best_ro_gain` and `$qubit.ext.best_ro_length` and, for a pulse readout with the needed calibration coordinates available, the resonator readout module and slot.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
