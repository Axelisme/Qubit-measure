# Qutrit reference populations

**Readers:** operator, agent. Acquire three labeled IQ clouds to calibrate a g/e/f classifier and estimate populations in preparation-label coordinates.

## What is measured

In each sequence, prepare g (no pulse), e (**g↔e π pulse**), or f (**Pi 03** → **Pi 32**), then **Readout** → **Relax delay**. There is no extra reset pulse. The three preparations are references, not guaranteed pure levels.

## Key settings

- Locate 0–1 and 0–3 with ground-state spectroscopy and calibrate the π pulses with Rabi. Prepare 3 with **Pi 03**, then locate 3–2 and calibrate **Pi 32** with [prepared-state amplitude Rabi](../../twotone/rabi/amp_rabi/guide.md#prepared-state-amplitude-rabi). For **Pi 32**, choose the configured charge-drive line, not a fast-flux line. Enter each calibrated RF frequency, gain, duration and generator. Missing transition settings leave required fields Invalid.
- Use a calibrated single-ADC **Readout**. For writeback, save these readout settings as a library module and bind its key before Run. Select it without changing non-classifier settings. For the integer-fluxonium simulation example, see [FTF reference settings](../../../simulation_envs/FTF/README.md#整數-fluxonium-的三態參考驗收).
- Set **Relax delay** to return to the ground preparation between shots. Use at least six **Shots per preparation**; Analysis requires all three complete IQ rows. It trains on the first half of each row and evaluates on the rest, including the extra shot when the total is odd.

## How to read the results

Live shows the three preparation IQ clouds. Analyze overlays the training centroids on the acquired clouds; the evaluation matrices use only held-out shots. Assignment uses the nearest centroid with no rejection region (exact ties choose g, then e, then f). The color maps and scatters have no shot SEM bars.

**Training confusion** has requested labels in rows and classified labels in columns. Analyze solves against the transposed matrix to estimate preparation-coordinate populations, without clipping negative or above-one values. Those departures can reflect finite shots or preparation contamination. The result is not pure-state fidelity.

Check the held-out macro and per-label errors together with the matrix condition number and smallest singular value. A large condition number or small singular value indicates sensitivity to errors in the correction. **Minimum centroid separation (shared σ)** is the closest pair's IQ separation divided by a pooled noise width. Do not accept a correction based on any one of these numbers alone.

## Next step

If the clouds overlap or the held-out error is high, check spectroscopy, preparation and the ADC window before using correction. Writeback replaces only the classifier on the captured readout module with the g/e/f centers and training confusion. If the target is missing, unresolved or differs apart from the classifier, bind the intended module and acquire a new Run. Changing readout or preparation does not automatically invalidate stored calibration; reacquire the references before using the new settings.

Use this calibration for g/e/f readout in [STIRAP](../../twotone/stirap/delay/guide.md) or [Raman](../../twotone/raman_gf/guide.md). The third outcome is f, not **Other**; experiments needing a rejected class use the GE circle classifier instead. This experiment has no MCP recipe.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
