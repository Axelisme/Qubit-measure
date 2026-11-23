**Readers:** operator, agent. Find two reset carriers from an averaged-IQ contrast map.

## Sequence

Optional **Reset** → optional **Preparation pulse** → simultaneous fixed-gain reset tones → **Readout** after the longer tone. The GUI sweeps both RF axes in hardware.

## Key settings

- Verify calibrated preparation (if used) and readout. **Tone 1 RF frequency** and **Tone 2 RF frequency** each start at `$qubit.ext.reset_f1` or `$qubit.ext.reset_f2` ± 50 MHz. Center each on its predicted sideband and use enough points to resolve the feature.
- Check both gains in **Tested reset**, seeded from `$qubit.ext.reset_gain1` and `$qubit.ext.reset_gain2`. **Relax delay** starts at 5 × T1 (150 µs if absent); enter the wait if T1 is invalid.

## Read the figures and choose carriers

Live and Analyze show the magnitude of mean IQ minus median background IQ, not population or reset fidelity. A host-stepped notebook Run also shows the latest acquired tone 2 row. No per-cell shot SEM is shown.

Analyze marks the RF pair picked after smoothing on the raw map. **Peak selection filter** chooses wavelet or Gaussian; **Filter strength** scales the wavelet threshold or sets Gaussian width in sample steps. If the star hits a boundary, shift that axis. If it is interior, tighten both windows until several cells resolve the peak.

The proposal updates only `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2`. Use the carriers in the gain and duration sweeps.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../../operator-workflow.md).
