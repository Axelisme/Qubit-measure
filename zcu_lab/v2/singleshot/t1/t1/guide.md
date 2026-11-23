# Single-shot T1

**Readers:** operator, agent. Measure relaxation from the g- and e-prepared population curves versus free evolution time.

## What is measured

Optional **Reset** → no drive for g preparation or **g↔e π pulse** for e → swept free evolution → **Readout**.

## Key settings

- Before Run, use Pulse **Readout** (not Mux) with a circle classifier and positive radius under **Advanced**. Obtain a missing radius from [GE Post-Analysis](../../ge/guide.md); linear cannot report Other. Calibrate the π pulse before comparing preparations.
- Set **Free evolution time** to capture the early drop and the late plateau. Its Stop and **Relax delay** start at 5 × stored ordinary T1 (150 µs if absent; blank if invalid). Reset does not shorten the relaxation wait automatically.
- **Sampling** `decay` (default) concentrates points early; `linear` spaces them evenly. The window and point count stay fixed. Enter a sweep, not a list of delays.

## How to read the results

Live shows raw g/e/Other fractions. Analyze uses only the selected result's saved confusion matrix for correction; without it, it shows classified fractions. Other includes shots outside both circles or tied and is not an identified energy level. Compare the g- and e-prepared traces against the rate-model fit. Reduced χ² compares residual size with the assumed errors; check for systematic residuals before using the fitted times. Bars show one shot SEM from the retained counts at each delay, estimating the sampling uncertainty of the class probabilities. Analyze plots and fits the linear estimates without clipping them to [0, 1]. It propagates shot-count covariance with the saved calibration matrix treated as known. Fewer than two retained shots give no bar. The bars do not show fit errors or round-to-round variation.

**Relaxation times (short to long)** lists the two model decay times in µs. Each is the inverse of the negative real part of a nonstationary mode's eigenvalue. The numbers are not separate lifetimes of the g and e preparations, and neither is singled out as qubit T₁. The times can be equal. For a complex-conjugate pair they describe the same e-folding time of the oscillation envelope.

If the saved matrix cannot produce finite corrected fractions, Analyze shows **Readout correction unavailable** and only the G/E/Other classified fractions from this Run, with shot SEM where available. These are counts after classification, before correction. There is no rate-model fit or decay time. When only a corrected bar is unavailable, the corrected point remains.

## Next step

Check the last points of both preparations. If either trace is still changing, extend Stop and keep enough points near the early drop. This run does not write T₁ back. Older files without class counts cannot be converted on Load; acquire again.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
