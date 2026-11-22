# Single-shot reset check

**Readers:** operator, agent. Compare populations with and without a tested reset across a qubit drive gain sweep.

## What is measured

Each independent branch starts with optional **Reset** and then applies one of three sequences before **Readout**: Rabi; Rabi → **Tested reset**; or Rabi → **Tested reset** → Rabi. The two Rabi pulses use the same swept gain.

## Key settings

- Use Pulse **Readout** (not Mux) with a calibrated circle and positive radius under **Advanced**. Use [GE Post-Analysis](../ge/guide.md) to obtain the radius. Linear cannot report Other.
- Include the intended π gain in **Qubit drive gain**. Stop starts at the smaller of 4 × stored π gain and 1.0. **Reps** are shots per gain per branch per round.
- **Relax delay** starts at 5 × ordinary T1 (150 µs if absent; blank if invalid). Adding reset does not shorten it automatically. Match preparation and wait conditions across comparisons.

## How to read the results

Compare the three panels at the same gain, especially the intended π gain. The reset-only panel reports mean and maximum sampled P(e); also check the maximum Other and the worst sampled gain. Other includes shots outside both circles or tied. Neither Other nor branch contrast establishes leakage or reset-channel fidelity.

Live shows raw fractions. Analyze uses only the selected result's saved confusion matrix for correction; files with counts but no compatible calibration show classified fractions. Bars show one shot SEM from the retained counts at each gain, estimating the sampling uncertainty of the class probabilities. Analyze propagates count covariance through any correction. Fewer than two retained shots give no bar. The bars do not show fit errors, histogram intervals or round-to-round variation.

## Next step

If P(e) is low in the no-tested-reset branch at the intended π gain, check the π pulse before judging reset. No calibration is written. Load cannot convert old files without counts; collect raw IQ separately if you need to refit readout.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
