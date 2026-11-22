**Readers:** operator, agent. Use repeated rotations to make small amplitude errors visible as alternating IQ contrast.

## Sequence and key settings

Optional **Reset** → X90 preparation → repeat the selected pulse for 0 to **Max repetitions** counts → calibrated **Readout**. Under **Repeat pulse**, one count adds one X180 or a pair of X90 pulses. Calibrate X90 and X180 with Rabi first and check readout. About ten repetitions can make a small error visible. **Relax delay** starts at `5 × qubit.t1` (150 µs if absent).

See [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md) for the shared GUI steps.

## Read the plot and decide what to tune

The trace shows the principal-axis mean IQ projection (a.u.), not population, against repetition count. Gaps mark unmeasured counts. The title uses the selected pulse if source cfg is available, and counts the largest retained-shot total per point when statistics are available. One error bar is one shot SEM: retained-shot sample SD along the plotted axis divided by √(retained shot count). It is not shot spread, round-to-round scatter or fit error. Mean-only files plot without bars.

Analyze redraws the trace from Run or Load, without fitting or writeback. Load alone does not open a plot. Look for alternating contrast that grows with repetition count. Preparation and drive history can also change the trace. Compare controls before changing pulses, then use the zig-zag gain or frequency sweep to locate a tuning minimum.
