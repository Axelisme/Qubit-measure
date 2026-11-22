**Readers:** operator, agent. Check how much a tested reset erases a Rabi rotation and whether contrast returns after a second pulse.

## Sequence

Optional **Reset** → **Rabi pulse** at the swept gain → one of three branches: readout; **Tested reset** then readout; or **Tested reset** then the same Rabi pulse and readout.

## Key settings

- Verify the qubit pulse, tested reset and readout. **Rabi pulse gain** replaces the pulse gain in both rotations. Use at least six finite points per branch and aim for at least one full period. With fitted frequency f in cycles/gain, cover 1/f and keep gain steps below 1/(4f) to resolve the 2f term.
- **Relax delay** starts at 5 × T1 (150 µs if absent); enter the wait if T1 is invalid. Leave the optional preparation reset disabled until calibrated.

## Read the figures and decide whether to verify the reset

Live and the three fit panels project mean IQ on the before-reset principal axis (I if unavailable). Bars are shot SEM: sample SD along that axis divided by √(retained shots), not fit error or round-to-round scatter. Mean-only files show no bars.

Analyze overlays fits using the before-reset frequency f; the reset-plus-Rabi fit adds 2f. Compare residual oscillation after reset with contrast recovered by the second pulse. The after/before amplitude ratio is relative IQ contrast, not fidelity. Phase is referenced to zero gain and may be unavailable when the fundamental is unresolved; this is not a zero offset.

**Fit residuals** shows measured minus fitted IQ; RMS gives the typical mismatch in a.u. Structured residuals suggest the model missed a feature, and the 2f term alone does not identify its cause. Confirm a promising setting with a population measurement. There is no writeback.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).
