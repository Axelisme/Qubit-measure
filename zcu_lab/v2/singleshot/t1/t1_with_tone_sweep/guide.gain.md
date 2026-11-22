# Single-shot T1 under a probe tone: gain sweep

**Readers:** operator, agent. Map driven transition rates versus probe gain, using both initial preparations.

## What is measured

For each gain and exposure time: optional **Reset** → no pulse for g preparation or **Pi Pulse** for e → **Probe Pulse** for the swept length → **Readout**. The joint fit evolves g/e/Other populations and yields transition rates in µs⁻¹. In the rate panels, Γ₀₁ and Γ₁₀ are g→e and e→g; Γ₀L and Γ₁L are rates into the Other class, not rates into a resolved energy level.

## Key settings

- Before Run, use Pulse **Readout** (not Mux) with a circle classifier and positive radius under **Advanced**. Obtain a missing radius from [GE Post-Analysis](../../ge/guide.md); linear cannot report Other. Calibrate the π pulse before comparing preparations.
- **Gain** starts at 0–1. Confirm the fixed **Probe Pulse** RF frequency. Analyze uses `$qubit.ext.ac_stark_coeff × gain²` when the Run captured a coefficient, even if the rate-panel axis still says **Probe gain (a.u.)**. Keep the outer grid coarse enough to repeat a full exposure sweep at each point.
- Set a finite, strictly positive **Probe length** Start. Stop starts at 5 × the stored tone T1, falling back to ordinary T1 and then 30 µs as the time scale. An invalid stored value leaves the required Stop blank. **Relax delay** is separate: it starts at 5 × ordinary T1 (150 µs if absent; blank if invalid) and is not shortened automatically by reset.
- **Sampling** `linear` (default) spaces the exposures evenly; `decay` concentrates points early without changing the window or point count.

## How to read the results

Live shows g/e/Other fractions for both preparations. Analyze drops incomplete outer rows, applies only the saved readout correction if available, and shows six population maps above the rate panels. Map colors do not show shot SEM. The log-scale rate bars (every fifth point) come from the rate-fit covariance, not shot SEM or run-to-run variation.

## Next step

If the populations have not approached a plateau, extend **Probe length** and keep enough early points to resolve the slope. Compare rates only where both preparations constrain the fit. No calibration is written. Older files without class counts cannot be converted on Load.

For general operations, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
