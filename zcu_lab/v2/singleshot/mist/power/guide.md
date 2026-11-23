# MIST probe-gain sweep

**Readers:** operator, agent. Find the onset of changes in single-shot populations as the probe gain increases.

## What is measured

Optional **Reset** → optional **Initial preparation pulse** → **MIST tone** at the swept gain → **Readout**. Other includes shots outside both decision circles or tied; it does not identify a leaked level.

## Key settings

- Before Run, use Pulse **Readout** (not Mux) with a circle and positive radius under **Advanced**. A linear classifier cannot report Other; obtain a missing radius from [GE Post-Analysis](../../ge/guide.md).
- Set the **MIST drive gain** window (initially 0–1) to sample below and above the expected onset. Keep preparation and tone duration fixed when comparing gains.
- **Relax delay** starts at 5 × ordinary T1 (150 µs if absent; blank if invalid). Adding reset does not shorten it automatically.
- Analyze converts gain to mean photon number with `$qubit.ext.ac_stark_coeff × gain²` when the Run captured that coefficient. Obtain it from [AC-Stark calibration](../../ac_stark/guide.md). **Logarithmic x-axis** changes the scale, not the measured probabilities.

## How to read the results

Live shows classified fractions. Analyze corrects them only with the confusion matrix saved with this result; without it, expect classified fractions, not corrected populations. Look for a change in P(Other) across the scan. Compare it against the bars, one shot SEM from counts retained at each point (absent below two shots). Analyze uses the linear correction without clipping, so estimates can be below 0 or above 1. It propagates shot-count covariance with the saved calibration matrix treated as known. The bars do not show fit errors or round-to-round variation.

If the saved matrix is singular or solving it gives a nonfinite value at a valid point, Analyze logs one warning and uses classified populations for the whole scan. An unavailable shot SEM bar does not by itself disable correction.

## Next step

Widen or recenter the scan if the onset is at a limit. No calibration is written. Load cannot convert older files without class counts; acquire again.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
