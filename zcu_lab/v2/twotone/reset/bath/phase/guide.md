**Readers:** operator, agent. Find opposite tomography π/2 phases for a fixed bath reset.

## Sequence

Optional **Reset** → optional **Preparation pulse** → simultaneous cavity and qubit tones → phase-swept tomography π/2 after the cavity tone → **Readout**.

## Key settings

- Use calibrated readout and verify the π/2 in **Tested reset**. Set the tested durations from the bath length scan. The cavity RF and gain start from `$qubit.ext.reset_bath_freq` and `$qubit.ext.reset_bath_gain`.
- Sweep **Tomography π/2 phase** across at least one full turn to see both extrema. **Relax delay** starts at 5 × T1 (150 µs if T1 is absent); enter the wait if T1 is invalid.

## Read the figures and choose a phase

Live and Analyze show mean IQ on its principal variation axis (I until two means are available). It is not population. Bars are shot SEM along that axis: sample SD divided by √(retained shots), not fit error or round-to-round scatter. Mean-only files show no bars.

Live connects the measured points. Analyze shows unconnected measured points and a solid fixed-period (360°) cosine fit. Its ± values are 1σ phase errors from the fit; dashed lines show equivalent phases nearest the scan midpoint. Set the π/2 phase to the chosen extremum modulo 360°; subtract the present phase and wrap the difference to the nearest turn for the correction. A flat fringe or unresolved extrema do not establish a state assignment. Verify g/e against independently prepared references.

Analyze proposes `$qubit.ext.reset_bath_max_phase` and `$qubit.ext.reset_bath_min_phase`; when the tested cfg and calibration are available from that Run, it also proposes the tested reset as `$qubit.reset.reset_bath` (maximum) and `$qubit.reset.reset_bath_e` (minimum), changing only π/2 phase, and an optional `$qubit.module.reset` selection (ground by default, editable to excited).

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../../operator-workflow.md).
