**Readers:** operator, agent. Find the cavity RF and gain with the strongest phase-dependent bath-reset response.

## Sequence

Optional **Reset** → optional **Preparation pulse** → simultaneous cavity and qubit tones → tomography π/2 after the cavity tone → **Readout**. Each cell adds 0°, 90°, 180° and 270° to the π/2 phase in **Tested reset**.

## Key settings

- Calibrate the qubit x90 and readout before Run. The editable π/2 offset starts at 90°; the saved x90 is not changed.
- **Cavity drive RF frequency** starts at resonator RF minus 0.8–1.2 times the qubit Rabi frequency (`$qubit.ext.rabi_freq`). Center the sweep on the expected sideband. Choose **Cavity drive gain** within the calibrated range.
- **Relax delay** starts at 5 × T1 (150 µs when T1 is absent). Supply the wait if T1 is invalid; enabling **Reset** does not shorten it.

## Read the figures and choose the next settings

Live counts acquired cells. Live and Analyze show |S180−S0|² + |S270−S90|² in a.u.²; S is mean complex IQ at each relative phase. This is tomography contrast, not reset probability, and the map shows no per-cell shot SEM.

The Analyze star marks the maximum after smoothing, on the raw map. **Peak selection filter** and **Filter strength** can move the star without changing the colors; strength scales the wavelet threshold or sets Gaussian width in sample steps. If the star is on an edge, shift that axis within the calibrated range and repeat.

The proposal updates only `$qubit.ext.reset_bath_gain` and `$qubit.ext.reset_bath_freq`. Use those settings for the bath duration and phase sweeps.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../../operator-workflow.md).
