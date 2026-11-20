**Readers:** operator, agent. Sweep Pump and Stokes detuning with independent envelopes and interpret saved-readout g/e/f estimates.

# STIRAP detuning

Open `New Tab → twotone → stirap → detuning`.
This experiment sweeps Stokes detuning in the outer hardware loop and Pump detuning in the inner hardware loop.
One program and one acquisition cover the full grid. The computer does not compile or acquire each coordinate separately.
Each shot overlaps the two drives, waits for both to finish, then reads out g/e/f.
It does not infer carriers from a simulator or tune them automatically.

## Before Run

Measure the Pump 0–3 and Stokes 3–2 carriers separately.
Connect them to different generators and verify the physical charge-drive wiring.
Pump and Stokes envelopes overlap, and one channel cannot play both arbitrary-waveform carrier frequencies at the same time.
Enter the measured carriers in MHz and signed gains between -1 and 1.
The carrier fields remain blank until you provide them.

Prepare g/e/f references with `singleshot/qutrit_population`.
Analyze those references and Apply the centroid3 classifier to the readout module.
Confirm the resonator slot and writeback target before Apply.
A circle classifier's Other label is not f and cannot replace centroid3.

The Form takes readout from `$resonator.module.readout` when that slot exists.
Without a slot, it starts from the readout template and still needs the readout wiring, frequency and calibration.
A readout slot alone does not make the experiment ready.
Measured drive carriers, gains, scan endpoints and point counts remain required.
The initial relaxation delay uses five times the captured qubit T1, with 30 us as the T1 fallback when missing.
Confirm this delay for the device before Run.

Source changes do not refill an open Form.
Refresh updates sources and choices while preserving input.
Reset captures new sources and may replace unsaved edits.
Reset or reopen after applying a new readout classifier.

## Select the envelopes and scan

Pump waveform and Stokes waveform each accept gauss, cosine, drag or arb.
Both start as Gaussian with sigma 0.05 us and length 0.3 us.
These are Form initial values, not model defaults or measured calibrations.
Select a waveform library key or an inline shape independently for each drive.
Set the selected shape's parameters in us, including length and Gaussian/DRAG sigma.
DRAG also uses its delta and alpha parameters.
Arbitrary envelopes take their duration from the selected sampled asset.
No common width or six-sigma length rule couples the drives.
If an envelope exceeds the generator's memory, the existing upload check reports the error.

Positive Stokes lead delays Pump by that many us.
Negative Stokes lead delays Stokes instead.
Different pulse lengths are allowed. Readout starts only after both timelines finish.

Set Stokes detuning and Pump detuning endpoints in MHz and the number of points on each axis.
Each detuning adds to its own measured carrier.
The plotted coordinates are the requested offsets. Actual pulse frequencies follow the board's hardware grid.
Shots sets the g/e/f classification count per scan point.
Relax delay follows each shot.

## Read the results

Run stores float64 counts with shape `(Nstokes, Npump, 3)` in g/e/f order.
The measurement heatmap shows the raw f fraction, with Pump detuning on x and Stokes detuning on y.
Unfinished points remain NaN.

Analyze uses only the readout classifier saved in that Run.
It shows six heatmaps, raw g/e/f above corrected g/e/f.
Correction is not clipped to the interval 0 to 1.
The summary contains the saved training confusion matrix, its condition number and its minimum singular value.
The summary and recipe report give `f_population_estimate`, the largest finite
corrected f population on this grid, and `pump_detuning_mhz` and `stokes_detuning_mhz`, offsets from the measured carriers.
Ties go to the first acquired point. If no finite value is available, the peak
and its coordinates are null. The estimate is unclipped and does not identify
a global optimum.
These are preparation-label estimates, not transfer fidelity or an intermediate-state measurement.
Nonfinite or empty counts and singular confusion matrices fail analysis.
There is no calibration writeback.

Save keeps counts and the selected envelope configurations in native `data.h5`.
The native experiment tag is `twotone/stirap/detuning`.
This adapter does not load the old combined STIRAP experiment tag.

## MCP recipe

Use `recipe_guide(adapter="twotone/stirap/detuning")` for this guide.
Call `stirap_detuning` with measured `pump_freq_mhz` and `stokes_freq_mhz`.
`pump_start_mhz`, `pump_stop_mhz`, `pump_n` set the inner scan.
`stokes_start_mhz`, `stokes_stop_mhz`, `stokes_n` set the outer scan.
Both point-count fields set Form n, which becomes cfg expts.
`pump_waveform_ref` and `stokes_waveform_ref` accept a library key or an exact Form option such as `Inline: cosine`.
Use ordered overrides to set the selected shape's scalar parameters, such as `pump_waveform.length`.
Omitted inputs retain the Form value or source. Explicit null goes to the field owner and may make a required field invalid.
The recipe checks readiness, runs, analyzes and saves once.
Analysis failure still saves the acquired counts. Run failure does not enter the Save finalizer.
The recipe does not ask for writeback.
