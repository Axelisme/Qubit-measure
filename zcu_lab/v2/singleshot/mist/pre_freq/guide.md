# MIST preparation-frequency sweep

**Readers:** operator, agent. Measure how a variable-frequency preparation changes populations under a fixed MIST drive.

## What is measured

Optional **Reset** → **Preparation pulse** at the swept RF frequency → **Optional π pulse** → fixed **MIST tone** → **Readout**. Unlike the MIST probe-frequency sweep, the MIST tone is not scanned. Other includes shots outside both decision circles or tied, not identified leakage.

## Key settings

- Before Run, use Pulse **Readout** (not Mux) with a circle and positive radius under **Advanced**. A linear classifier cannot report Other; obtain a missing radius from [GE Post-Analysis](../../ge/guide.md).
- **Preparation RF frequency** starts at qubit frequency ± 1.5 × qubit linewidth, at 51 points. The **MIST tone** starts at resonator frequency. Check both pulse gains and durations; enable **Optional π pulse** only for the intended preparation, using a calibrated π pulse.
- **Relax delay** starts at 5 × ordinary T1 (150 µs if absent; blank if invalid). Adding reset does not shorten it automatically.

## How to read the results

Live shows classified fractions. Analyze corrects them only with the confusion matrix saved with this result; without it, expect classified fractions, not corrected populations. Later library or form edits do not alter that result's correction. Compare the three traces versus preparation RF frequency; this peak-location spectrum does not draw per-point shot SEM bars.

## Next step

If the feature touches a frequency limit, recenter at its RF frequency and extend until the neighboring baseline is visible. No calibration is written. Load cannot convert older files without class counts; acquire again.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
