**Readers:** operator, agent. Find candidate qubit transitions as fast-flux pulse gain changes.

# Fast-flux two-tone spectroscopy

After optional Reset, play Flux pulse and the frequency-swept Qubit probe in parallel, then read out after both finish. This changes the pulsed bias, not the DC setting.

## Choose the scan

- Bind qubit and resonator. Check calibrated **Readout** and the **Qubit probe** drive channel, length and gain. The **Qubit probe frequency** scan supplies the probe RF.
- Check the **Flux pulse** channel, timing and allowed gain. **Flux pulse gain** overrides the module gain and starts at one zero-gain point. Set an approved signed range for a flux dependence; a.u. is DAC full-scale fraction, not flux.
- Set **Qubit probe frequency** in absolute MHz. The initial window is the captured qubit center ± 1.5 linewidths. Widen it to cover the expected transition shift. Choose a **Relax delay** long enough for repeatable preparation.

For tab setup, refreshing sources and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

## Read the map and choose a follow-up

Live and Analyze share the gain–RF heatmap. Color is mean complex IQ projected along the scan's largest IQ variation, in a.u., not a population. There is no shot SEM. A valid saved map can still plot without historical cfg.

Follow a continuous ridge to locate a candidate transition at the gain of interest. If it hits the boundary, extend that RF endpoint without coarsening the grid. For overlapping bands, narrow the gain window and compare with lower probe gain to check power broadening. The map does not identify or fit a transition and does not write back calibration.
