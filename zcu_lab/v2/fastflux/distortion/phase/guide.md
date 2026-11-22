**Readers:** operator, agent. Scan a two-pulse phase probe across a flux transient and tell a phase response from loss of fringe contrast.

# Fast-flux local phase

Open `fastflux/distortion/phase`. After optional Reset, start three branches together: the flux pulse, a wait followed by a pair of π/2 pulses, and Minimum readout wait. The second π/2 pulse follows the first with no extra free-evolution delay. Sweep its carrier phase. Read out after all branches finish. Unlike the accumulated-phase experiment, both π/2 pulses in this test move with the wait.

## Open and run

Bind the intended qubit and resonator before Open. Defaults captures the current sources and discards edits. Reset uses the qubit reset slot and can be disabled. π/2 pulse and Readout use the calibrated x90 and resonator readout slots when present, otherwise editable templates. Use a calibrated π/2 pulse at the working flux.

Flux pulse uses the qubit fast-flux channel and starts with a 1 µs rectangular envelope at 0 MHz, gain 1.0 a.u. and Nyquist zone 2. Set a confirmed channel and safe gain. Its pre-delay places the leading edge of the flux pulse. Its waveform duration places the trailing edge. Check pulse delays and readout integration against the chosen wiring.

- π/2 pair wait starts at 0 to 1 µs, 101 points. Include data before both flux edges and after the tail.
- Projection phase starts at −180 to 180 deg, 61 points. Use several distinct phases spanning a cycle. It is the second pulse's carrier phase.
- Minimum readout wait starts at 2 µs. Set it to cover the latest pulse end and desired flux settling if the readout time must stay fixed.
- Reps is shots per point each round, initially 1000. Rounds is the number of full scans averaged, initially 10. Relax delay starts at five times captured T1, or 150 µs when T1 is unavailable.

## Read the maps and phase response

The Run and first Analyze heatmaps share the same axes. The π/2 pair midpoint is the swept wait plus one π/2 pulse duration. Projection phase is in deg. Color is the real projection of mean complex IQ along the scan's largest IQ variation, in a.u. This average has no shot classification and does not give a state population. The Run title counts points with IQ. Use round progress to tell how many full maps have been averaged.

The Phase response plot shows the phase extracted from the sine and cosine coefficients across each fringe, unwrapped along time. The dotted markers are phase estimates, not the original IQ samples. The green rectangular reference uses the programmed flux interval and median levels from the lowest and highest fifths by absolute extracted phase. Gray bands cover one π/2 pulse length on either side of each edge. Within the bands, the pair samples across the edge and limits time resolution. The 2D scan has no pointwise shot SEM. The derived phase markers have no fit uncertainty or quality threshold.

Inspect the fringe contrast at each time before interpreting its phase. Low contrast can make the estimated phase jump even when the line did not. If the phase remains offset at the right edge despite usable contrast, extend π/2 pair wait past that edge. For example, add 0.2 µs and 20 points to a 0.01 µs grid without coarsening it. Compare the offset to the pre-pulse baseline before tuning compensation. This plot does not estimate a transfer function or propose a correction.

Save retains the waiting and phase coordinates, complex IQ and source cfg. The time axis is saved in seconds and shown in µs. Load leaves the current form unchanged. Analyze needs the saved cfg for pulse timing. This experiment has no recipe or calibration writeback.
