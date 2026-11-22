**Readers:** operator, agent. Run the readout-chain length Rabi from the GUI or MCP and read its live curves, joint fit and residuals.

# K-cycle readout-chain length Rabi

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Behavior

`chain/len_rabi` sweeps the Drive pulse length like `twotone/rabi/len_rabi`, but every repetition is one cycle of 16 reads instead of one read.
A cycle waits Cycle thermalisation wait K, then plays eight blocks of the length-swept Drive pulse followed by two reads of the same Readout, with no reset inside the cycle.
Use at least 5 × T1 for K, including before the first cycle, to start near thermal equilibrium.
After a readout pulse ends, its `post_delay` sets the wait before the second readout or the next Drive pulse.
For cavity ring-down, start with a wait of 5/(2πκ), with κ the readout linewidth in MHz and the result in µs. The fallback readout template leaves 0.015 µs of arriving drive on both sides of the ADC window.
The length loop runs outside the cycle. Reps sets the cycles at every length in one round, and Rounds sets the number of rounds.

The host reduces each cycle to 65 numbers: the IQ of the 16 reads, their I², IQ, Q² summed over three read groups, and IQ products between read pairs summed over six groups set by the number of Rabi pulses and reads between them.
For a fixed sweep, the saved sums and the accumulator across rounds have fixed size, independent of the number of cycles.
Temporary arrays within one acquisition round grow with points × Reps. Increasing Reps therefore uses more temporary memory; increasing Rounds does not enlarge those per-round arrays.

## Parameter sources

The Drive pulse starts from a fresh `qub_probe` const pulse with gain 1.0, not from x180, because x180 usually has a shaped waveform that the FPGA length sweep cannot stretch.
Its gain is an uncalibrated starting value; confirm it before Run.
The length sweep accepts const and flat_top drive waveforms.
Readout uses the `$resonator.module.readout` slot or a fresh `readout` template, and is a single-tone pulse readout, not mux.
The length runs from 0.03 µs to 4 × the x180 length (1.0 µs without x180) over 101 points. The cycle has no Reset or Init Pulse.
Cycle thermalisation wait K starts at 5 * `$qubit.t1` µs, or 150 µs when T1 is missing.
Reps starts at 100 and Rounds at 100. Reps must be at least 66, so that even a single length gives the first round 65 cycles beyond its mean to estimate the noise of the 65 numbers.
One cycle lasts about Cycle thermalisation wait K + 8 × (pulse length + 2 × readout length).
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

Open this experiment with the qubit and its resonator bound in the selected point. Choose or expand the pulse and readout modules, check Drive duration › Start, Stop and Expts, the number of sampled values, then Run. In `linrange`, enter endpoints and a point count. `center` uses a midpoint and span instead of endpoints. The `_step` variants use spacing instead of a point count. Missing inputs must be filled before Run.
Frequency is the RF carrier (MHz). Leave Mixer freq blank when no mixer setting is needed. Gen ch selects the drive output and NQZ its Nyquist zone. Gain, when shown, is the signed DAC full-scale fraction (a.u.). Phase sets the rotation axis in degrees. The waveform style and Length set the envelope. Pre-delay waits before playback and Post-delay waits after playback, both in µs.
For Readout, use the resonator's calibrated RF frequency, gain and phase. RO ch selects the receiver. RO length sets its integration window and Trig offset places that window, both in µs. An optional classifier stores state-calibration data but this sequence does not classify or filter shots.

## Run and analyze

During Run, the upper live panel shows the mean IQ of the first read in each cycle. The I and Q weights follow the direction of greatest variation across the sweep; the projected signal is in a.u., not a classified state probability. With too few points to estimate that direction, the plot uses I. The lower panel combines the 65 features into a joint projection. It subtracts the across-sweep median and scales by noise estimated within each sweep point. Its unit is one point-mean standard error. It helps you see whether the joint data also resolve the oscillation; there are no shot-SEM bars on this pooled, noise-normalized diagnostic. The header advances once per completed round and shows cycles accumulated per point. Stop retains completed rounds.

The first Analyze figure (`fit`) repeats the first-read live points (blue) and adds the seed model (orange). The vertical bars on both first-read plots are one **shot SEM**: the projected single-cycle sample standard deviation divided by the square root of the retained cycles at that point. Each cycle contributes its first read once. The bars come from the saved I, Q, I², IQ and Q² sums, not from round-to-round scatter or the uncertainty of a fitted parameter.
The damped cosine is the first-read fit. The dashed π and dotted π/2 markers give candidate durations when they lie within the scan. The first-read π duration starts the joint model at f = 1/(2 × first-read π duration).
In the first-read figure, R² is the fraction of variation explained by the fit; the normalized residual RMS divides residual RMS by the signal range. A negative R² means the curve fits worse than a constant at the mean. A smaller normalized residual RMS indicates closer agreement. Neither metric sets a selection threshold. The traces need enough contrast and sweep coverage to distinguish the fit from noise.

The `joint residuals` figure compares the 65 measured statistics at each duration with the joint model. The upper panel uses the same direction, centering and scale as the lower live panel. Blue points are measurements and the solid line is the model. The lower panel shows each measured projection minus the model at that duration, in the same point-mean SEM units. Use the zero line to see where the model over- or underpredicts. These projected values are not state populations and have no per-point shot SEM bars.

The joint fit uses a nondecaying sin²(π f × duration) shape. It gives π duration = 1/(2f) and marks it with a dashed line when it falls within the scan. Treat it as a total π duration only for a constant waveform. Flat-top ramps contribute rotation area that the zero-phase model does not account for. The reported π duration and Rabi frequency errors are Wald standard errors, not shot SEM. R = 1/(2π × sweep span × frequency standard error) describes frequency precision.

The figure and summary also report the joint whitened residuals as numbers. J is the sum of their squares. The nominal degrees of freedom df are 65 times the number of durations minus the joint parameter count. Compare J/df with the approximate chi-square reference band [1 − 2√(2/df), 1 + 2√(2/df)]. The |r| > 3 fraction counts the whitened components with magnitude above 3. Those components use model covariance and are different from the signed residual points in the figure. Look for structure across durations before interpreting the numbers. None of these diagnostics labels or rejects the fit. The Analyze summary identifies the run and reports the saved Reps×Rounds setting, not the number of rounds that finished.

## What to do next

If both fits locate a compatible π duration within the scan, centre a dedicated length Rabi calibration there. For example, if a constant 0.10 µs pulse gives a credible π candidate at 0.13 µs, the candidate is 0.03 µs longer at the same gain. Extend the range if the first extremum is missing and add cycles when the shot SEM hides the oscillation. Use a separate calibration for a flat-top π duration.
This experiment does not propose writeback. The model assumes independent thermalised cycles, Markov transitions between three states and three shared IQ centers with one isotropic noise width. A read that changes the state places its IQ between the initial and final centers. This does not identify multiple jumps within a read, cavity transients or a physical leakage level. Large Wald errors or negligible thermal population of a state can leave its dynamics poorly determined. Fit convergence or a small residual is not proof of unbiased parameters.

Layout reference: Ding, *Novel Gates with Superconducting Fluxonium Qubits*, Fig. 5-13(b) (printed p. 157), separates measured points from a dense fit; it plots benchmarking survival against sequence length rather than projected IQ against drive duration.

## Saved data and MCP recipe

Native Save retains the drive duration axis, per-point cycle count and 65 feature sums, and the 65 × 65 product sums pooled across points for the live joint projection. Load keeps the same data and can reproduce the analysis; the first-read shot SEM uses moments already present in these files.

`chain_len_rabi` can set Drive pulse and Readout library references, the drive RF frequency and fixed gain, Drive duration › Stop/Expts, Reps and Rounds. Use an override on `relax_delay` for Cycle thermalisation wait K. Omitted inputs keep the form values, overrides apply last, and null is passed to the form as an explicit clear where accepted. The recipe runs, analyzes and saves without writeback. It reports π duration, Rabi frequency and their standard errors alongside the first-read π duration, R and the joint residual numbers. The per-duration points stay in the figure, not in the summary or recipe report.
