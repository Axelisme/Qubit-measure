**Readers:** operator, agent. Find a duration for two simultaneous reset tones from the averaged-IQ response.

## Sequence

Optional **Reset** → optional **Preparation pulse** → both reset tones → **Readout** after the longer tone.

Use different generator channels when the reset tones overlap. The compiler rejects two overlapping ordinary `Pulse` playbacks on the same generator, even on separate timelines.
See the [Program timeline rules](../../../../../../lib/zcu_tools/program/v2/README.md#原生寫法與具名-timeline) and the [`TwoPulseReset` timing rules](../../../../../../lib/zcu_tools/program/v2/modules/README.md#原生寫法).

## Key settings

- Check preparation (if used) and readout. In **Tested reset**, use calibrated carriers and gains; the starting values are `$qubit.ext.reset_f1`, `$qubit.ext.reset_f2`, `$qubit.ext.reset_gain1` and `$qubit.ext.reset_gain2`.
- Sweep **Tone 1 pulse duration** through the expected plateau. Tone 2 duration = swept duration + the configured tone 2 minus tone 1 duration. Set **Start** so both are positive.
- **Relax delay** starts at 5 × T1 (150 µs if absent); enter the wait if T1 is invalid.

## Read the figures and choose a duration

Live and Analyze show mean IQ on its principal variation axis (I until two means are available). It is contrast, not population. Bars are shot SEM: sample SD along that axis divided by √(retained shots), not fit error or round-to-round scatter. Mean-only files show no bars.

Choose the shortest acquired duration on a sustained plateau within the point errors. Extend **Stop** if the trace is still changing; refine around the first transition while keeping the tone duration difference. Enter the chosen duration manually and verify the state preparation separately. There is no writeback.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../../operator-workflow.md).
