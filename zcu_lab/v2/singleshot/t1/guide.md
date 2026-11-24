# Single-shot T1

**Readers:** operator, agent. Measure relaxation from the g- and e-prepared population curves versus free evolution time.

## What is measured

Optional **Reset** → no drive for g preparation or **g↔e π pulse** for e → swept free evolution → **Readout**.

## Key settings

- Before Run, use Pulse **Readout** (not Mux) with a circle classifier and positive radius under **Advanced**. Obtain a missing radius from [GE Post-Analysis](../ge/guide.md); linear cannot report Other. Calibrate the π pulse before comparing preparations.
- Set **Free evolution time** to capture the early drop and the late plateau. Its Stop and **Relax delay** start at 5 × stored ordinary T1 (150 µs if absent; blank if invalid). Reset does not shorten the relaxation wait automatically.
- **Sampling** `decay` (default) concentrates points early; `linear` spaces them evenly. The window and point count stay fixed. Enter a sweep, not a list of delays.

## How to read the results

Live shows raw g/e/Other fractions. Analyze uses only the selected result's saved confusion matrix for correction; without it, it shows classified fractions. Other includes shots outside both circles or tied and is not an identified energy level. Compare the g- and e-prepared traces against the rate-model fit. Reduced χ² compares residual size with the assumed errors; check for systematic residuals before using the fitted times. Bars show one shot SEM from the retained counts at each delay, estimating the sampling uncertainty of the class probabilities. Analyze plots and fits the linear estimates without clipping them to [0, 1]. It propagates shot-count covariance with the saved calibration matrix treated as known. Fewer than two retained shots give no bar. The bars do not show fit errors or round-to-round variation.

**T1** is the decay time of the mode with the larger absolute amplitude in the fitted e population for an ideal e preparation. **T1b** is the other mode's decay time. If the amplitudes are equal, Analyze uses the longer time for T1 and logs the tie. The times are not separate lifetimes for the two preparations. A complex-conjugate pair gives the same envelope decay time for both modes.

If the saved matrix is singular or solving it gives a nonfinite value at a valid point, Analyze logs one warning and uses classified populations for the whole scan. Analyze still fits both preparations. An unavailable shot SEM bar does not by itself disable correction.

## Next step

Check the last points of both preparations. If either trace is still changing, extend Stop and keep enough points near the early drop. Writeback proposes T1 in µs at `$qubit.t1`. Use Apply to store it.

Older files without class counts cannot be converted on Load; acquire again.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
