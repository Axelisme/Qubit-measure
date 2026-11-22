**Readers:** operator, agent. Sweep the bath-reset duration and find where its phase-dependent IQ contrast settles.

## Sequence

Optional **Reset** → optional **Preparation pulse** → simultaneous cavity and qubit tones → π/2 at 0°, 90°, 180° or 270° absolute after the cavity tone → **Readout**.

## Key settings

- Calibrate readout and verify the π/2 in **Tested reset**. The cavity RF and gain start from `$qubit.ext.reset_bath_freq` and `$qubit.ext.reset_bath_gain`; use settings checked in the bath frequency sweep.
- Set **Qubit-tone duration** to span the rise and a plateau. The cavity duration follows it with the configured cavity-minus-qubit duration difference held fixed. Keep both durations positive.
- **Relax delay** starts at 5 × T1 (150 µs if T1 is absent). If T1 is invalid, enter the wait. The sweep repeats for each **Rounds** value, and each point within a pass also acquires that many rounds.

## Read the figures and choose a duration

The I and Q panels show four phase branches versus qubit-tone duration (µs). Their bars are shot SEM: sample SD in that quadrature divided by √(retained shots), not fit error or round-to-round scatter. Mean-only files show no bars. With unequal shot counts across partial passes, means weight the passes equally and no pooled SEM is shown.

The contrast panel shows |S180−S0|² + |S270−S90|² (a.u.²), with S the complex phase-branch mean. It has no direct shot SEM and is not a population or reset fidelity. Compare the acquired points; a flat trace with no resolved response does not establish a plateau.

Choose a duration once contrast has settled after a resolved change; extend **Stop** if contrast is still changing. Enter the duration manually, then verify state preparation in the bath phase step. There is no writeback.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../../operator-workflow.md).
