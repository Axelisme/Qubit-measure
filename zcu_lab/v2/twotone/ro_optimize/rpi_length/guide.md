# R-π readout window sweep

**Readers:** operator, agent. Select a readout window using the alternating IQ contrast of repeated reads and π pulses.

## Sequence and key settings

One **Reps** count is a cycle of 64 readout → π pairs. There is no reset, relax delay or discarded warmup. Lags 1–15 stay within each cycle. The next cycle can start only after the final ADC window closes.

- Calibrate **Qubit π pulse** first; a bad rotation changes every diagnostic. Set **ADC window** to span the useful integration range. Run rounds the window to the ADC clock and sets pulse duration to that window plus **Length margin**, rounded to the generator clock. The finite nonnegative margin starts at the captured pulse duration minus ADC window (template difference, normally 0.03 µs, if no module). Supply it if the lengths are unavailable. Verify that the pulse covers the window at the receiver with the chosen trigger offset.
- In **Readout pulse and ADC window**, check coverage and keep successive ADC windows disjoint. R → π waits for the readout's `post_delay` plus the π pulse's `pre_delay`, measured from the generator pulse end, not ADC closure. There is no separate wait field. Start readout `post_delay` at `5/(2πκ)` µs for linewidth κ in MHz, then compare a longer wait. To calibrate the delay separately, compare herald → nominal Xπ at both waits.

## Read the figures and choose a setting

Live plots **Alternating-contrast score** for settings with completed data. Invalid settings are gaps and counted separately; Stop keeps only completed rounds, and fails if none finish.
- **Alternating contrast** keeps the valid Live points, marks the highest score with a red dashed line and plots reasons for each invalid setting below.
- **Correlation diagnostics** shows effective retention Qeff, amplitude-to-noise ratio q, alternating amplitude a and one-quadrature noise variance σ². Qeff includes π error: Qeff = Q·(1−2ε) for symmetric π flip error ε. The ratio q = a/σ² is not d/σ. These diagnostics are not additional ranking scores.

Score = (T − 3.719√V₀)/r₀, where T is alternating lag contrast, V₀ is its white-noise null variance and r₀ is half the same-read variance. The gate is score > 0.0224 per setting. The square-root term is a penalty, not an error bar. There are no pointwise bars on the score or correlation diagnostics. The saved lag sums lack the variances and cross-lag covariances of the individual read products needed to estimate uncertainty. Correlated noise can pass the gate; the score is not Fisher information, readout QND or leakage.

A failed gate (`gate_failed`), Qeff outside [0,1], nonpositive alternating amplitude, noise variance or same-read variance, or a zero denominator excludes that setting. If no setting is valid, Analyze reports none rather than selecting a default. Check nearby settings for consistent diagnostics and repeat with an extended range if the winner is at a boundary. There is no writeback. Enter the selected ADC window and set pulse duration to that window plus the margin used in the Run. Older matching data can still be analyzed when its cfg cannot be restored.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
