**Readers:** operator, agent. Measure the qubit's 1/e relaxation time T1 versus fast-flux gain.

# Fast-flux T1

Each shot is optional Reset → calibrated π pulse → Flux pulse at the scanned gain and duration → Readout. Each gain has its own duration trace. The observable is mean IQ, not state population; gain needs separate calibration to become flux.

## Set up the scan

- Bind qubit and resonator. Check the **π pulse** (x180) and **Readout** calibration. Verify the fast-flux channel and permitted gain. The two scans override the module gain and duration; use a waveform with sweepable duration, such as const or flat_top.
- **Flux pulse gain** starts at one zero-gain point. Enter a signed DAC range in a.u. to measure gain dependence. **Flux pulse duration** starts at 0.1 µs and ends at five times captured T1; **Relax delay** starts at the same upper limit. Without T1 the limit is 150 µs; invalid stored T1 leaves the derived values blank. Keep durations positive and include the baseline at long times.

For tab setup, refreshing sources and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

## Read the decay and fit diagnostics

- Live and the first Analyze heatmap show mean IQ projected along the largest scan variation. The map has no shot SEM.
- Analyze refits each gain's exponential with a shared baseline from preliminary usable fits. **Rejected fits** compares fitted span with half the measured span and the mean absolute smoothed residual. A point at or below either bound fails; this is not an R² cutoff.
- **Fitted T1** omits rejected values. Its error bars are standard errors of the fitted 1/e time, not errors of mean IQ or variations across rounds. R² compares the fixed-baseline fit with the mean predictor; normalized residual RMS gives the typical mismatch as a fraction of measured span. Those diagnostics include rejected gains. A valid map can still be analyzed without historical cfg.

Check the decay approaches its baseline. If it does not, extend the window toward three fitted lifetimes while preserving enough early points for fast decays. Repeat before selecting a gain when neighboring lifetimes differ by less than their fit errors. There is no writeback.
