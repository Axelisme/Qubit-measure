# MIST gain–frequency map

**Readers:** operator, agent. Find regions where single-shot populations change with the MIST tone's gain and RF frequency.

## What is measured

Optional **Reset** → optional **Initial preparation pulse** → **MIST tone** at the selected gain and frequency → **Readout**. Other includes shots outside both decision circles or tied; it does not identify a leaked level.

## Key settings

- Before Run, use Pulse **Readout** (not Mux) with a circle and positive radius under **Advanced**. A linear classifier cannot report Other; obtain a missing radius from [GE Post-Analysis](../../ge/guide.md).
- **MIST drive RF frequency** starts at resonator frequency ± 1.5 × stored resonator linewidth. **MIST drive gain** starts at 0–1. Cover the line and the expected onset without sacrificing resolution along either axis.
- **Relax delay** starts at 5 × ordinary T1 (150 µs if absent; blank if invalid). Adding reset does not shorten it automatically.

## How to read the results

Compare g/e/Other map panels at the same gain and RF frequency on their common 0–1 scale. Live shows classified fractions. Analyze corrects them only with the confusion matrix saved with this result; without it, expect classified fractions, not corrected populations. An elevated Other region is a loss-of-classification signal, not a level assignment. The maps carry no per-point shot SEM bars.

## Next step

Extend the corresponding axis if the region runs to a scan edge. No calibration is written. Load cannot convert older files without class counts; acquire again.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
