# Single-shot amplitude Rabi

**Readers:** operator, agent. Sweep qubit drive gain and measure Rabi oscillations from single-shot IQ.

## What is measured

Optional **Reset** → **Qubit drive pulse** at the swept gain → **Readout**. Run retains raw IQ. Analyze jointly fits the Rabi oscillation and the g/e readout distributions, with a constant oscillation amplitude and phase fixed at zero. The period gives the π gain (half a period) and π/2 gain (a quarter period), measured from zero gain.

## Key settings

- Scan **Qubit drive gain** over a full oscillation. It starts at −0.3; Stop starts at the smaller of 1.0 and 4 × stored π gain. For a stored π pulse with scalar length, the editable drive copy starts at `1.5 × π length × π gain`. Invalid length or gain can leave length blank. An absent π pulse starts the probe at gain 1 and length 1 µs; verify both. With a stored arbitrary waveform, select a different waveform to edit duration.
- Use a single ADC: Pulse or one-tone Mux **Readout**. Raw acquisition and fitting work without a classifier, but Live population preview then stays off. For preview, use a circle classifier with radius under **Readout → Advanced**; linear has no Other, and a circle without radius requires [GE Post-Analysis](../ge/guide.md).
- **Initial State** describes the predominant state before the drive, not the state at the first sampled gain. **Reps** are shots per gain per round; **Rounds** add shots to each point. **Relax delay** starts at 5 × ordinary T1 (150 µs if absent; blank if invalid). Reset does not shorten it automatically.

## How to read the results

Use Live to check population contrast and Analyze to compare class fractions with the joint-fit curves and locate π and π/2. Bars are one shot SEM of the retained class fractions, absent with fewer than two shots. They are not fit errors, histogram widths or round-to-round variation.

Use the projected IQ histogram and fitted confusion matrix to check g/e separation. Negative log-likelihood is the fit objective, not a population error or acceptance threshold. The matrix's Other row is fixed by the model; Other is a rejected class, not an identified leakage level.

## Next step

If the scan misses a full oscillation, widen it before relying on the fitted gains. A valid joint fit with finite calibration can propose a full circle classifier on the bound readout module (on the sole tone for Mux); it does not write the drive gain. The measured readout settings must match the captured module apart from its classifier.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
