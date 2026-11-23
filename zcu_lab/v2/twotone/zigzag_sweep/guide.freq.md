**Readers:** operator, agent. Sweep the repeated pulse's RF frequency to find the carrier with the least variation across zig-zag repetitions.

## Sequence and key settings

At each frequency: optional **Reset** → X90 preparation → repeat the selected X180 or X90 pair from zero to **Max repetitions** → calibrated **Readout**. Calibrate X90/X180 and check readout first. Start with a small count such as six to limit acquisition time.

**Repeated gate RF frequency** starts at `qubit.freq ± 2 MHz`. Center it on the measured line for the pulse being tuned. **Relax delay** starts at `5 × qubit.t1` (150 µs if absent). In Analyze, **Search from** and **Search to** restrict the interval used to pick the minimum.

See [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md) for the shared GUI steps.

## Read the plots and refine the carrier

Run and Analyze map projected mean IQ (a.u.), not population, versus repetition count and repeated-pulse RF frequency in MHz. The Analyze curve is the smoothed sum of absolute IQ differences between adjacent counts. It measures flatness, not a separate IQ measurement or gate fidelity. The map and derived loss have no pointwise shot SEM.

The minimum marker and `best_value` identify a candidate within the search interval. If the minimum sits at an edge, shift or widen the search before using it. If it is interior, scan more finely around it and confirm with an independent trace. Update only the intended pulse or qubit frequency after review; there is no writeback.
