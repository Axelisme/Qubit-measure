**Readers:** operator, agent. Sweep the repeated pulse's gain to find the gain with the least variation across zig-zag repetitions.

## Sequence and key settings

At each gain: optional **Reset** → X90 preparation → repeat the selected X180 or X90 pair from zero to **Max repetitions** → calibrated **Readout**. Calibrate X90/X180 and check readout first. Start with a small count such as six to limit acquisition time.

**Repeated gate gain** starts at `0.8 × stored X180 gain … 1.2 × stored X180 gain`; recenter on the X90 gain when repeating X90 pairs. **Relax delay** starts at `5 × qubit.t1` (150 µs if absent). In Analyze, **Search from** and **Search to** restrict the interval used to pick the minimum.

See [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md) for the shared GUI steps.

## Read the plots and refine the gain

Run and Analyze map projected mean IQ (a.u.), not population, versus repetition count and gain of the repeated gate. The Analyze curve is the smoothed sum of absolute IQ differences between adjacent counts. It measures flatness, not a separate IQ measurement or gate fidelity. The map and derived loss have no pointwise shot SEM.

The minimum marker and `best_value` identify a gain candidate within the search interval. If the minimum sits at an edge, shift or widen the search before using it. If it is interior, scan more finely around it and check the repeat trace. There is no writeback; update only the intended pulse after review.
