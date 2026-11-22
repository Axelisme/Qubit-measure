# Single-shot length Rabi

**Readers:** operator, agent. Sweep the qubit drive duration and measure Rabi oscillations from single-shot IQ.

## What is measured

Optional **Reset** → **Qubit drive pulse** at the swept duration → **Readout**. Each duration has one cloud of **Shots per length** raw IQ values. Analyze jointly fits the oscillation and the g/e readout distributions; π and π/2 lengths are half and a quarter of the fitted period, respectively.

## Key settings

- Use a waveform with editable length; native arbitrary waveforms have asset-defined duration and are rejected before scanning. **Drive envelope length** starts at 0.03 µs and ends at 4 × stored π length. If no scalar π length is stored, Stop starts at 1 µs; an invalid stored length leaves Stop blank. The selected π pulse keeps its gain, or the probe starts at gain 1 if there is no stored π pulse. Verify gain and scan at least a full oscillation.
- Use one ADC: Pulse or one-tone Mux **Readout**. Raw acquisition and fitting need no classifier; Live population preview requires a circle with positive radius under **Advanced**. Linear has no Other; obtain a missing radius from [GE Post-Analysis](../ge/guide.md).
- **Initial State** is the predominant state before the drive, even if the first length is nonzero. **Fit phase offset** allows a phase within ±90° of that preparation (off by default); the fitted zero-length population can then differ from the pre-pulse population. **Relax delay** starts at 5 × ordinary T1 (150 µs if absent; blank if invalid). Reset does not shorten it automatically.

## How to read the results

Look for oscillations in Live and compare each measured population curve with the joint fit in Analyze. Bars show one shot SEM from the retained class counts at each duration. They estimate the sampling uncertainty of that point's class probabilities. Fewer than two retained shots give no bar. The bars do not show histogram widths, fit errors or round-to-round variation.

Check the projected IQ histogram and fitted confusion matrix for separation. Negative log-likelihood is a fitting objective, not an acceptance threshold. Other is a rejected class, and its matrix row is fixed by the model.

## Next step

Extend the sweep if π is beyond the measured range or there is no full oscillation. A valid joint fit with finite calibration can propose a complete circle classifier on the bound readout module (on the sole tone for Mux); it does not write the drive length. The measured readout settings must match that captured module apart from its classifier.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
