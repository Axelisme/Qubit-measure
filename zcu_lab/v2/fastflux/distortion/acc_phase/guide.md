**Readers:** operator, agent. Use a moving Ramsey preparation and fixed projection to infer qubit detuning during a flux transient.

# Fast-flux accumulated phase

Open `fastflux/distortion/acc_phase`. After optional Reset, run the flux pulse, a wait followed by the first π/2 pulse, and Minimum projection wait in parallel. Once all three branches finish, play the second π/2 pulse at the swept projection phase and read out. The first pulse moves through the transient; keep the second pulse at a fixed time so the phase slope can measure the frequency shift.

## Set pulses and scans

Bind the qubit and resonator before Open. Defaults refreshes these sources and drops edits. Reset uses the qubit reset slot and is optional. The π/2 pulse uses the calibrated x90 slot and Readout uses the resonator readout slot if available. Otherwise enter the calibrated settings in the shown templates.

Flux pulse starts from the qubit fast-flux channel with a 1 µs rectangular envelope, 0 MHz carrier, gain 1.0 a.u. and Nyquist zone 2. Check that the channel and allowed gain are correct. Pre-delay sets the programmed start, and the waveform length sets the duration. Set the readout gain and integration timing from the resonator calibration.

| Field | Initial value and effect |
| --- | --- |
| First π/2 wait | 0 to 2 µs, 101 points. Covers the initial baseline, the pulse and the return to baseline. |
| Projection phase | 0 to 360 deg, 21 points. The second π/2 pulse uses these carrier phases. Use distinct phases over a cycle to resolve a fringe. |
| Minimum projection wait | 2 µs. Increase it to at least the later of the flux branch completion and the largest wait plus the first π/2 branch duration. Include pulse delays. Otherwise the projection time moves at the end of the scan and the phase slope does not have the intended meaning. |
| Reps, Rounds, Relax delay | 1000 shots per point per round, 10 full scans averaged, and five times the captured T1 between shots. If T1 is unavailable, start with 150 µs. |

## Read the map and detuning

The Run and first Analyze heatmaps share the first π/2 pulse center on x, second-pulse projection phase on y, and the same IQ projection. The center is First π/2 wait plus half the pulse duration. Color is mean IQ projected along the largest variation in the scan, in a.u., not a classified state population. The Run title counts finite IQ points. The acquisition progress separately reports how many whole maps contribute to the average.

For each time slice, Analyze fits the sine and cosine fringe coefficients and unwraps their phase across the time scan. The detuning plot shows the negative slope of that phase divided by 2π. A positive slope of phase in rad/µs gives a negative detuning in MHz. Black points are numerical detuning estimates, not individual shots. The green rectangular reference has the programmed start and end times, with levels inferred from the lowest and highest deciles of absolute detuning. The gray bands each span one π/2 pulse duration centered on a programmed edge. There is no pointwise shot SEM on the 2D map or fit parameter error on the numerical derivative.

A washed-out fringe, a phase jump near one full turn, or a noisy derivative makes a feature unreliable. Sample fast enough that phase changes by less than π between adjacent first-pulse times. For a possible 25 MHz detuning that means less than 0.02 µs between samples. If the estimate remains outside the acceptable frequency offset at the scan end, extend the wait range at that spacing and increase Minimum projection wait accordingly. Do not infer a compensation filter from the derivative alone.

Save stores waits in seconds, projection phase in degrees, complex IQ and source cfg. Load leaves the current form unchanged. Analyze needs the saved cfg for the pulse timing. No recipe or calibration writeback is published for this experiment.
