# Single-shot T1 under a probe tone

**Readers:** operator, agent. Measure relaxation while a fixed-frequency, fixed-gain tone is on.

## What is measured

Optional **Reset** → optional **Initial preparation pulse** → no π pulse for g preparation or **g↔e π pulse** for e → **Probe tone** for the swept exposure time → **Readout**. The fitted slow T₁ and fast T₁ (`T1_b`) are the inverse rates of the two nonstationary relaxation modes under the tone.

## Key settings

- Before Run, use Pulse **Readout** (not Mux) with a circle classifier and positive radius under **Advanced**. Obtain a missing radius from [GE Post-Analysis](../../ge/guide.md); linear cannot report Other. Calibrate the π pulse before comparing preparations.
- Check **Probe tone** frequency and gain. **Probe exposure time** is the tone length, not an idle delay. Stop starts at 5 × the stored tone T1, falling back to ordinary T1 and then 30 µs as the time scale. An invalid stored value leaves the required Stop blank. **Relax delay** is separate: it starts at 5 × ordinary T1 (150 µs if absent; blank if invalid) and is not shortened automatically by reset.
- **Sampling** `decay` (default) concentrates points early; `linear` spaces them evenly without changing the window or point count. Enter a sweep, not a list of durations.

## How to read the results

Live shows raw g/e/Other fractions versus exposure. Analyze uses only the selected result's saved confusion matrix for correction; without it, it shows classified fractions. Other includes shots outside both circles or tied and is not an identified energy level. Compare the g- and e-prepared traces against the rate-model fit. Reduced χ² compares residual size with the assumed errors; check for systematic residuals before using the fitted times. Analyze plots and fits the linear estimates without clipping them to [0, 1]. Bars show one shot SEM from the retained counts (at least two shots), propagated with the saved calibration matrix treated as known. They do not show fit errors or round-to-round variation.

If the saved matrix cannot produce finite corrected fractions, Analyze shows **Readout correction unavailable** and only the G/E/Other classified fractions from this Run, with shot SEM where available. These are counts after classification, before correction. There is no rate-model fit or decay time. Writeback is unavailable for that analysis. When only a corrected bar is unavailable, the corrected point remains.

## Next step

If the longest exposure misses the plateau, extend Stop before accepting the result. Writeback proposes the slow T₁ under the tone in µs at `$qubit.ext.t1_with_tone`. Older files without class counts cannot be converted on Load; acquire again.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
