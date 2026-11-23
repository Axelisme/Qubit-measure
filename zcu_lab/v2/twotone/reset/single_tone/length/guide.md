**Readers:** operator, agent. Find the duration of a single-tone sideband reset from the averaged-IQ response.

## Sequence

Optional **Reset** → optional **Preparation pulse** → single reset pulse at fixed gain and carrier → **Readout**.

## Key settings

- Verify preparation (if used) and readout. Check the gain and carrier in **Tested reset**; its RF starts from the calibrated `$qubit.ext.reset_f`.
- Set **Reset pulse duration** to cover the onset and plateau. **Relax delay** starts at 5 × T1 (150 µs if absent); enter the wait if T1 is invalid.

## Read the figures and choose a duration

Live and Analyze show mean IQ projected onto its principal variation axis (I until two means are available). Compare the change with duration, not the sign. This is not population or fidelity. Bars are shot SEM: sample SD along that axis divided by √(retained shots), not fit error or round-to-round scatter. Mean-only files show no bars.

Choose the earliest measured duration on a sustained plateau within the point errors. Extend **Stop** if the trace is still changing; lower **Start** or refine the steps if it starts on the plateau. Enter the duration in the tested pulse or saved reset module yourself; there is no writeback. Confirm the state with a separate preparation/readout check.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../../operator-workflow.md).
