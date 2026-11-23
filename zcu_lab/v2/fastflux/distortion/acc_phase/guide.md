**Readers:** operator, agent. Measure qubit detuning during a fast-flux transient from the slope of the accumulated Ramsey phase.

# Fast-flux accumulated phase

After optional Reset, start three branches together: Flux pulse, wait → first π/2 pulse, and Minimum projection wait. After all three finish, play the second π/2 pulse at the scanned phase and read out. Move the first pulse through the transient and keep the second pulse time fixed.

## Set the timing

- Bind the qubit and resonator. Check calibrated **π/2 pulse** and **Readout** settings (initially x90 and the resonator readout slot when available). Check the fast-flux channel and allowed gain. **Flux pulse** pre-delay and waveform length set the programmed edges.
- Cover the initial baseline, pulse and tail with **First π/2 wait**. Sample densely enough that phase changes by less than π between adjacent times: for 25 MHz, use a step below 0.02 µs. Scan distinct **Projection phase** values across a full cycle.
- Set **Minimum projection wait** at least as long as both flux branch completion and the longest First π/2 wait plus its pulse branch, including delays. A shorter setting lets the projection time move and changes the inferred slope.
- **Relax delay** starts at five times the captured T1, or 150 µs without T1.

For tab setup, refreshing sources and saving, see [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md).

## Read the map and detuning

Live and the first Analyze heatmap show first π/2 center (wait + half π/2 duration) versus Projection phase. Color is mean IQ projected along the scan's largest IQ variation, not population.

Analyze unwraps the sine/cosine fringe phase across time. The black line connects detuning estimates, not a fit. The points are `−(dφ/dt)/(2π)` in MHz for phase in rad and time in µs. A positive phase slope means negative detuning. The green rectangle uses the programmed flux edges and levels from the lowest and highest deciles of absolute detuning. Each gray band spans one π/2 duration centered on an edge. Each cell stores only mean IQ, without the shot covariance needed to estimate phase uncertainty and propagate it through the time derivative. The detuning points have no error bars. Analyze requires the source cfg for pulse timing.

Check fringe contrast before trusting a jump or noisy derivative. If detuning still exceeds your tolerance at the right edge, extend the wait range at the same spacing and increase Minimum projection wait. The derivative alone does not determine a compensation filter. There is no writeback.
