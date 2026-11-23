# Thermal-chain readout window sweep

**Readers:** operator, agent. Rank window settings by the correlations across a chain of readouts, without a calibrated π pulse.

## Sequence and key settings

Each shot makes 16 reads, without drive or reset. Only the readout's timing separates reads. **Thermal wait K** follows each shot and starts at 5 × T1; set it long enough for thermal equilibrium. The first shot starts in the pre-Run state. The measurement needs thermal g/e contrast, not the population reached under repeated readout.

- Set **ADC window** to span the useful integration range. Run sets the pulse duration to the window plus **Length margin**. The margin must be finite and nonnegative. It starts at the captured pulse length minus ADC window (template difference, normally 0.03 µs, if no module). Enter the margin if either length is unavailable. Verify the selected trigger offset and that the pulse covers the ADC window at the receiver.
- Use **Readout pulse and ADC window** to set timing. A new pulse can start before the previous ADC window closes; keep windows on the same ADC disjoint. Start readout `post_delay` at `5/(2πκ)` µs for linewidth κ in MHz, then compare a longer wait. To calibrate the delay separately, compare herald → nominal Xπ at both waits.

## Read the figures and choose a setting

Live and Analyze plot the same three panels versus window (µs):
- **Thermal-chain score** combines first-pair correlation ρ and modeled amplification G, with a penalty for positive correlation growth M: ρ²·G·[|ρ|/(|ρ|+max(M,0))]²; zero at ρ = 0. The dashed line marks the highest score; exact ties keep acquisition order.
- **Estimated QND** uses the decay of the early cross covariances. It is not clipped to [0,1]. Check the trade-off at the selected setting.
- **Correlation diagnostics** compares M and |ρ|. M can rise even with two-state redistribution; it is not a leakage or MIST probability. The score is neither calibrated g/e SNR nor a direct measure of information.

The score and QND bars are approximate model-conditional error scales, not shot SEM or confidence intervals. They hold M and the main IQ axis fixed. The selection uses the score only, not its error bars. Stop keeps only settings with a completed round; stopping before any do causes Run to fail.

Extend a boundary peak and repeat. There is no writeback. Enter the selected ADC window and set pulse duration to that window plus the margin used in the Run. Older matching data can be reanalyzed from saved moments even when its cfg cannot be restored.

## MCP recipe

`ro_optimize_chain_length`: `length_start_us`, `length_stop_us` and `points` set **ADC window › Start, Stop, Expts**. Omit `readout_ref`, those three inputs, `relax_delay`, `reps` or `rounds` to keep the Form value. Explicit null does not keep it: `readout_ref: null` is rejected by the required module selector; null numeric edits pass to the field owner, and unfilled required fields fail the readiness check before Run. Overrides apply before that check. The recipe saves completed Run data even if Analyze fails.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
