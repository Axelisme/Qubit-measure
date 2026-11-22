# MIST probe-frequency sweep

**Readers:** operator, agent. Locate changes in single-shot g/e/Other probabilities versus the MIST tone's RF frequency.

## What is measured

Optional **Reset** → optional **Initial preparation pulse** → **MIST tone** at the swept frequency → **Readout**. Other includes shots outside both decision circles or tied; it does not identify a leaked level.

## Key settings

- Before Run, use Pulse **Readout** (not Mux) with a circle and positive radius under **Advanced**. A linear classifier cannot report Other; obtain a missing radius from [GE Post-Analysis](../../ge/guide.md).
- **MIST drive RF frequency** starts at resonator frequency ± 1.5 × stored resonator linewidth. Include off-feature points on both sides. Set **Reps** and **Rounds** to sample each probability sufficiently.
- **Relax delay** starts at 5 × ordinary T1 (150 µs if absent; blank if invalid). Adding reset does not shorten it automatically.

## How to read the results

Live shows classified fractions. Analyze corrects them only with the confusion matrix saved with this result; without it, expect classified fractions, not corrected populations. Compare the three traces on the same RF axis to locate resonances or loss features. The lines join measured fractions, not fits. Bars show ±1 shot SEM from the g/e/Other counts at each point, not fit error or round-to-round scatter. Analyze uses the linear correction without clipping, so estimates can be below 0 or above 1. It propagates shot-count covariance with the saved calibration matrix treated as known. Fewer than two shots or missing counts leave no bar.

## Next step

If a feature hits a frequency limit, center the next sweep on its RF frequency and extend the window until neighboring baselines are visible. No calibration is written. Load cannot convert older files without class counts; acquire again.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
