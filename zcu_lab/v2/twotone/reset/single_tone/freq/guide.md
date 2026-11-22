**Readers:** operator, agent. Locate the sideband carrier of a single reset pulse from the averaged-IQ spectrum.

## Sequence

Optional **Reset** → optional **Preparation pulse** → frequency-swept reset pulse at fixed gain and duration → **Readout**.

## Key settings

- Verify preparation (if used) and readout. **Reset drive RF frequency** starts at `$qubit.ext.reset_f` ± 1.5 × `$qubit.ext.reset_fwhm`. Widen the window if the center is only a rough estimate.
- Check **Tested reset** gain and duration. For cavity ringdown, start its `post_delay` at 5/(2πκ) µs for resonator linewidth κ in MHz. For readout timing, calibrate with herald → nominal Xπ and compare a longer wait.
- **Relax delay** starts at 5 × T1 (150 µs if absent); enter the wait if T1 is invalid.

## Read the figures and choose the carrier

Live counts acquired RF points. Both plots show mean IQ on its principal variation axis (I until two valid means are available). The contrast is in a.u., not excited-state probability; there is no shot SEM.

Analyze overlays the Lorentzian and its center. The center and FWHM ± values are 1σ fit errors. Confirm measured points on both sides of the feature. Recenter a clipped feature and use center ± 1.5 × FWHM once resolved. Reduce the step if the fitted width is about one frequency step.

The proposal updates `$qubit.ext.reset_f` and `$qubit.ext.reset_fwhm`; when the tested cfg and calibration are available from that Run, it also proposes `$qubit.reset.reset_10` with only the frequency changed and an optional `$qubit.module.reset` selection. Sweep duration separately.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../../operator-workflow.md).
