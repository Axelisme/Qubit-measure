**Readers:** operator, agent. Choose the gains of two simultaneous reset tones using the IQ contrast at calibrated carriers.

## Sequence

Optional **Reset** → optional **Preparation pulse** → both reset tones → **Readout** after the longer tone.

## Key settings

- Calibrate readout and verify preparation if enabled. Check both carriers in **Tested reset**, seeded from `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2`.
- Set **Tone 1 gain** and **Tone 2 gain** over their usable calibrated ranges. **Relax delay** starts at 5 × T1 (150 µs if absent); enter the wait if T1 is invalid.

## Read the figures and choose gains

Both maps show the magnitude of mean IQ relative to the minimum-gain corner, not population or fidelity. There is no per-cell shot SEM. Analyze marks a smoothed-map extremum with a star on the raw map. Its choice of extremum depends on the first scan corner versus the map mean; reversing an axis can change that corner.

**Peak selection filter** and **Filter strength** control the smoothing used for selection, not the displayed colors. Strength scales the wavelet threshold or sets Gaussian width in sample steps. Check an edge candidate on a nearby repeat and verify state preparation separately.

The proposal updates `$qubit.ext.reset_gain1` and `$qubit.ext.reset_gain2`; when the tested cfg and calibration are available from that Run, it also proposes `$qubit.reset.reset_120` with only those gains changed and an optional `$qubit.module.reset` selection.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../../operator-workflow.md).
