**Readers:** operator, agent. Measure relaxation during a flux pulse and decide which gain-dependent T1 estimates can be used.

# Fast-flux T1

Open `fastflux/t1`. After optional Reset, excite the qubit with a calibrated π pulse, apply the flux pulse with the selected gain and duration, then read out. One row of the measurement varies flux pulse duration at a fixed gain. The observable is mean complex IQ; converting it to state population or the gain to flux requires separate calibration.

## Set up the scan

Bind qubit and resonator before Open. Defaults refreshes the captured sources and discards edits. Reset uses the optional reset slot. π pulse and Readout use the calibrated x180 and readout slots, or editable templates when the slots are absent. Check the readout integration timing.

Set the Flux pulse channel, carrier and timing for the selected fast-flux line. The template uses the qubit fast-flux channel and starts with a 0.1 µs rectangular waveform at 0 MHz and Nyquist zone 2. The two sweeps override gain and duration. Choose a waveform with sweepable duration, such as const or flat_top.

Flux pulse gain starts at one point at zero gain. Enter the signed DAC gain range in a.u. to measure a gain dependence. Flux pulse duration initially runs from 0.1 µs to five times captured T1 with 101 points. Use strictly positive durations. Relax delay starts at the same upper limit; missing T1 uses a 30 µs time scale and gives 150 µs, while a stored invalid T1 leaves the derived fields empty. Reps starts at 1000 shots per point per round, Rounds at 10 full sweeps.

## Read the figures

Run and the first Analyze figure use the same flux gain/duration heatmap. Color is mean IQ projected along the largest scan variation, in a.u. It is not a state population. The live title counts cells with acquired IQ; round progress shows the number of scans contributing to the average.

Analyze fits an exponential at each gain, finds the median baseline among preliminary fits passing its contrast checks, and refits each trace with that baseline fixed. It accepts a fit when the fitted span is greater than half the measured span and greater than the mean absolute smoothed residual. Those tests reject flat fits and fits whose mismatch overwhelms their decay contrast. They are not R² thresholds. If a fit is rejected, the Rejected fits figure shows the fitted IQ span and the two bounds at that gain. A marker at or below either bound explains the rejection.

The Fitted T1 figure shows the 1/e time at each gain and the fit standard error. Rejected T1 values are not plotted. R² reports how much of the measured variance the fixed-baseline fit explains; it can be negative if a constant at the data mean describes the observations better. The normalized residual RMS is residual RMS divided by the measured IQ span. A residual RMS of 0.10 means the typical mismatch is one tenth of the trace span. These panels show diagnostics even at gains whose T1 was rejected. The 2D map has no shot SEM. The error bar on a T1 estimate is a fit parameter error, not a standard error of the mean IQ.

Check that each usable decay approaches its baseline within the window. If the candidate T1 is 40 µs and the scan currently ends at 40 µs, try an end near 120 µs before comparing small changes in lifetime. Keep enough early points to resolve the fast decays as well. If neighboring T1 estimates differ by less than their fit errors, repeat before choosing a gain. This experiment does not write back a lifetime or choose an operating point.

Save and Load preserve gain, duration, IQ and source cfg. Duration is saved in seconds and shown in µs. A valid saved map can be analyzed without historical cfg; the acquisition summary then omits Reps×Rounds. Load does not restore the current form. There is no recipe for this experiment.
