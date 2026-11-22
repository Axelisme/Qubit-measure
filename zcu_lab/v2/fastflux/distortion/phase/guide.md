**Readers:** operator, agent. Move a pair of π/2 pulses through a flux transient to separate a local phase response from fringe contrast loss.

# Fast-flux local phase

After optional Reset, start Flux pulse, wait → π/2 → π/2, and Minimum readout wait together. Sweep the second pulse's carrier phase without an extra gap between the pulses. Read out after all branches finish.

## Set pulses and scans

- Bind the qubit and resonator. Check a calibrated **π/2 pulse** and **Readout** at the working flux. Check the **Flux pulse** channel and allowed gain. Pre-delay sets the leading edge; waveform length sets the trailing edge.
- Set **π/2 pair wait** to cover the flux edges, pre-pulse baseline and tail. Cover one cycle with distinct **Projection phase** values. Set **Minimum readout wait** beyond the latest pulse end and desired settling if readout time must stay fixed.
- **Relax delay** starts at five times captured T1, or 150 µs without T1.

For tab setup, refreshing sources and saving, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).

## Read the maps and phase response

Live and the first Analyze heatmap use the pair midpoint (wait + one π/2 duration) and Projection phase. Color is mean IQ projected along the scan's largest IQ variation, not state population. Acquired points and completed rounds track coverage and averaging separately.

Analyze extracts the fringe phase and unwraps it along time. The black line connects phase estimates, not a fit. The green rectangle compares the programmed interval with levels from the lowest and highest fifths by absolute extracted phase. Gray bands span one π/2 duration on either side of each edge, where the pair can sample across the edge. The 2D map has no shot SEM and the phase estimates have no fit errors or automatic quality threshold. Analyze requires the source cfg for pulse timing.

Check fringe contrast before trusting a phase jump. If usable contrast remains but phase has not returned to baseline, extend **π/2 pair wait** without coarsening the steps. The plot does not estimate a transfer function or write back compensation.
