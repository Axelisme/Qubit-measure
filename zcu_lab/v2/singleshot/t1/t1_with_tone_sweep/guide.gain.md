# Single-shot T1 under a probe tone: gain sweep

**Readers:** operator, agent. Map driven transition rates versus probe gain, using both initial preparations.

## What is measured

For each gain and exposure time: optional **Reset** → no pulse for g preparation or **Pi Pulse** for e → **Probe Pulse** for the swept length → **Readout**. The joint fit evolves g/e/Other populations and yields transition rates in µs⁻¹. In the rate panels, Γ₀₁ and Γ₁₀ are g→e and e→g; Γ₀L and Γ₁L are rates into the Other class, not rates into a resolved energy level.

## Key settings

- Before Run, use Pulse **Readout** (not Mux) with a circle classifier and a positive radius under **Advanced**. If the radius is missing, obtain it from [GE Post-Analysis](../../ge/guide.md#next-step) and apply it before Run. Linear cannot report Other. Calibrate the π pulse before comparing preparations.
- **Gain** starts at 0–1. Confirm the fixed **Probe Pulse** RF frequency. Analyze uses `$qubit.ext.ac_stark_coeff × gain²` when the Run captured a coefficient, even if the rate-panel axis still says **Probe gain (a.u.)**. Keep the outer grid coarse enough to repeat a full exposure sweep at each point.
- Set a finite, strictly positive **Probe length** Start. Stop starts at 5 × the stored tone T1, falling back to ordinary T1 and then 30 µs as the time scale. An invalid stored value leaves the required Stop blank. **Relax delay** is separate: it starts at 5 × ordinary T1 (150 µs if absent; blank if invalid) and is not shortened automatically by reset.
- **Sampling** `linear` (default) spaces the exposures evenly; `decay` concentrates points early without changing the window or point count.

## How to read the results

Run shows six population maps (preparation by column, final class by row) versus probe gain and duration. The bottom two panels connect the measured fractions versus probe duration at the current gain, one per preparation.

Analyze drops incomplete outer rows and applies the saved readout correction if available. Its six population maps put g and e preparations in separate rows, and final g/e/Other in columns. Color shows population, not shot SEM. The bottom panels compare Γ₀L with Γ₀₁, Γ₁L with Γ₁₀, then Γ₁L with Γ₀L. Lines join rates fitted separately at each sweep point on log-scale axes. Bars at the first and every fifth point after it are the rates' fit SE from their covariance, not shot SEM or run-to-run variation. The Analyze maps and rates use the same outer coordinate; if transformed as above, do not read it as raw gain.

## Next step

If the populations have not approached a plateau, extend **Probe length** and keep enough early points to resolve the slope. Compare rates only where both preparations constrain the fit. No calibration is written. Older files without class counts cannot be converted on Load.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
