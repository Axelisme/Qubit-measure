**Readers:** operator, agent. Run the STIRAP Pump-by-Stokes gain sweep and interpret saved-readout g/e/f maps.

Open `twotone/stirap/amplitude` in measure-gui, or call MCP recipe `stirap_amplitude`.
This experiment sweeps Pump gain and Stokes gain independently.
Run acquires the two-dimensional grid in one round.

## Before Run

- Measure the Pump 0–3 and Stokes 3–2 carrier frequencies at the intended operating point. Enter both in MHz. Do not substitute simulator parameters for measured carriers.
- Bind Pump and Stokes to separate charge-drive lines. Both play during the overlap.
- Select a pulse readout with a saved `centroid3` calibration. The calibration must contain distinct g/e/f IQ centers and a training confusion matrix. Obtain it separately before this experiment.
- Set the two gain ranges in a.u. Enter signed endpoints within [-1, 1].

After changing aliases, component parameters or the module library, use Defaults to seed the Form from a new capture.
Defaults may overwrite unsaved input. Refresh only updates sources and choices; it keeps the current input.

## Controls and sequence

Pump and Stokes each accept gauss, cosine, drag or arb. They can use different families and lengths.
The Form initially selects Gaussian envelopes with sigma 0.05 µs and length 0.3 µs.
For arb, choose an available waveform asset. Its samples determine the duration.

`stokes_lead` is in µs. A positive value starts Stokes first and delays Pump.
A negative value starts Pump first. Zero starts both together.
The program joins both pulse ends before the terminal readout.
It uses the same two envelopes, carrier frequencies and lead at every gain pair.

`pump_gain.start`, `pump_gain.stop` and the Pump point count define the outer gain sweep.
`stokes_gain.start`, `stokes_gain.stop` and the Stokes point count define the inner gain sweep.
Both axes offer Range or Center coordinates, with Expts or Step sampling.
The displayed axes record the requested gains.
`shots` sets repetitions at each gain pair in one acquisition round.
`relax_delay` is the wait in µs after each shot.

## Analyze and Save

Run returns float64 g/e/f shot counts with shape `(number of Pump gains, number of Stokes gains, 3)`.
Save writes those counts with both requested gain axes.
The live figure shows three raw preparation-label fraction maps, one each for g, e and f.
Pump gain is the horizontal axis. Stokes gain is the vertical axis. Both use a.u.
To read a fixed Pump-to-Stokes ratio, follow a straight line through the origin on these maps.
An interrupted acquisition without a complete round leaves counts as NaN.

Analyze uses only the readout calibration saved with this Run.
The top row shows raw g/e/f maps. The bottom row shows confusion-corrected g/e/f maps without clipping.
The title gives the training confusion matrix's condition number and minimum singular value. A high condition number and a small minimum singular value warn that readout correction amplifies variations in the measured counts. These 2D maps do not show pointwise shot SEM.
The summary and recipe report give `f_population_estimate`, the largest finite
corrected f population on this grid, and `pump_gain` and `stokes_gain` in a.u..
Ties go to the first acquired point. If no finite value is available, the peak
and its coordinates are null. The estimate is unclipped and does not identify a global optimum. The cross on the corrected f map marks that sampled pair. Look for a broad high-f region with little corrected e fraction, not an isolated bright cell. If a candidate peaks at the upper Pump limit, extend that limit by one or two steps within the approved range, then scan the stable region at finer spacing.
Missing calibration, incomplete or empty counts, and singular confusion prevent analysis.
A large condition number warns that correction amplifies uncertainty. Corrected fractions can fall outside [0, 1].
These are preparation-label estimates, not transfer fidelity or a measurement of the intermediate state.

The MCP recipe performs Run, Analyze, then Save. It saves completed Run data even if Analyze fails.
A failed Run does not enter Analyze or Save. The recipe proposes no writeback.
For waveform selection, use `pump_waveform_ref` and `stokes_waveform_ref` with a library key or an inline Form option such as `Inline: cosine`.
Use `overrides` to set that waveform's parameters.
Recipe `pump_start/pump_stop/pump_n` and `stokes_start/stokes_stop/stokes_n` set the independent gain grids.
Each `n` maps to its Form point count and core `expts`.
