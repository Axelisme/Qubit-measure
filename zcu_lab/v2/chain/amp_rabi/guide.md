**Readers:** operator, agent. Run the readout-chain gain Rabi from the GUI or MCP and read its live curves, joint fit and checks.

# K-cycle readout-chain amplitude Rabi

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Behavior

`chain/amp_rabi` sweeps the Drive pulse gain like `twotone/rabi/amp_rabi`, but every repetition is one cycle of 16 reads instead of one read.
A cycle waits Cycle thermalisation wait K, then plays eight blocks of the gain-swept Drive pulse followed by two reads of the same Readout, with no reset inside the cycle.
Use at least 5 × T1 for K, including before the first cycle, to start near thermal equilibrium.
After a readout pulse ends, its `post_delay` sets the wait before the second readout or the next Drive pulse.
For cavity ring-down, start with a wait of 5/(2πκ), with κ the readout linewidth in MHz and the result in µs. The fallback readout template leaves 0.015 µs of arriving drive on both sides of the ADC window.
The gain loop runs outside the cycle. Reps sets the cycles at every gain in one round, and Rounds sets the number of rounds.

The host reduces each cycle to 65 numbers: the IQ of the 16 reads, their I², IQ, Q² summed over three read groups, and IQ products between read pairs summed over six groups set by the number of Rabi pulses and reads between them.
For a fixed sweep, the saved sums and the accumulator across rounds have fixed size, independent of the number of cycles.
Temporary arrays within one acquisition round grow with points × Reps. Increasing Reps therefore uses more temporary memory; increasing Rounds does not enlarge those per-round arrays.

## Parameter sources

The Form uses the `twotone/rabi/amp_rabi` sources: the x180 copy or a fresh `qub_probe` pulse, the `$resonator.module.readout` slot or a fresh `readout` template, and 51 gains from -0.3 to `min(1.0, 1.2 * x180 gain)`.
Select a single-tone pulse readout, not a mux module. The cycle has no Reset or Init Pulse.
Cycle thermalisation wait K starts at 5 * `$qubit.t1` µs, or 150 µs when T1 is missing.
Reps starts at 100 and Rounds at 100. Reps must be at least 66, so that even a single gain gives the first round 65 cycles beyond its mean to estimate the noise of the 65 numbers.
One cycle lasts about Cycle thermalisation wait K + 8 × (pulse length + 2 × readout length).
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

Open this experiment with the qubit and its resonator bound in the selected point. Choose or expand the pulse and readout modules, check Drive gain › Start, Stop and Expts, the number of sampled values, then Run. In `linrange`, enter endpoints and a point count. `center` uses a midpoint and span instead of endpoints. The `_step` variants use spacing instead of a point count. Missing inputs must be filled before Run.
Frequency is the RF carrier (MHz). Leave Mixer freq blank when no mixer setting is needed. Gen ch selects the drive output and NQZ its Nyquist zone. Gain, when shown, is the signed DAC full-scale fraction (a.u.). Phase sets the rotation axis in degrees. The waveform style and Length set the envelope. Pre-delay waits before playback and Post-delay waits after playback, both in µs.
For Readout, use the resonator's calibrated RF frequency, gain and phase. RO ch selects the receiver. RO length sets its integration window and Trig offset places that window, both in µs. An optional classifier stores state-calibration data but this sequence does not classify or filter shots.

## Run and Analyze

During Run, the upper live panel shows the mean IQ of the first read in each cycle. The I and Q weights follow the direction of greatest variation across the sweep; the projected signal is in a.u., not a classified state probability. With too few points to estimate that direction, the plot uses I. The lower panel combines the 65 features into a joint projection. It subtracts the across-sweep median and scales by noise estimated within each sweep point. Its unit is one point-mean standard error. It helps you see whether the joint data also resolve the oscillation; there are no shot-SEM bars on this pooled, noise-normalized diagnostic. The header advances once per completed round and shows cycles accumulated per point. Stop retains completed rounds.

The first Analyze figure (`fit`) repeats the first-read live points (blue) and adds the seed model (orange). The vertical bars on both first-read plots are one **shot SEM**: the projected single-cycle sample standard deviation divided by the square root of the retained cycles at that point. Each cycle contributes its first read once. The bars come from the saved I, Q, I², IQ and Q² sums, not from round-to-round scatter or the uncertainty of a fitted parameter.
The solid cosine is the first-read fit. The dashed π and dotted π/2 markers give the candidate gains when they lie within the scan. The first-read π gain seeds the joint fit at f = 1/(2 × first-read π gain).
In the first-read figure, R² is the fraction of variation explained by the fit; the normalized residual RMS divides residual RMS by the signal range. A negative R² means the curve fits worse than a constant at the mean. A smaller normalized residual RMS indicates closer agreement. Neither metric sets a selection threshold. The traces need enough contrast and sweep coverage to distinguish the fit from noise.

The `populations` figure tests whether a single physical model explains all 65 features across the sweep. Each panel shows one model-inferred first-read state population, not a classifier count. The points allow the operation to vary independently by sweep point. Their bars are one **Wald standard error**, taken from the local fit covariance, not shot SEM. The solid lines use the joint fit. The panel title gives the fitted thermal population before the drive. The plotted populations are after the drive. The y axes autoscale separately.
The joint fit uses sin²(π f × gain), with free state-population endpoints. Its π gain is the first positive half-period, 1/(2f); the dashed line marks it if inside the sweep.
State numbers are fit labels, not independently identified levels. R is 1/(2π × sweep span × fringe-frequency standard error), using the Rabi frequency in inverse sweep units. It is a frequency-precision diagnostic, not a pass/fail test. The joint weighted residual J is reported with its degrees of freedom (df). The separate checks report their own J/df and give p where available:

- **within-point** asks whether a free three-state model explains the data at each point.
- **sharing** checks whether sharing readout parameters worsens the fit relative to fitting points separately. It has no p-value.
- **curve** checks the constrained sweep model against a free operation at each point.

Compare the points and the model before interpreting the numbers. Large J relative to df or a small reported p indicates a mismatch to investigate; no check selects a calibration automatically. Missing diagnostics are not a pass. The Analyze summary identifies the run and gives its saved Reps×Rounds setting, not the number of rounds that finished.

## What to do next

If both fits resolve an oscillation and give compatible π gains, use that range to plan a dedicated amplitude Rabi calibration at the same drive frequency and duration. For a current x180 gain of 0.60 and a reliable candidate of 0.55 a.u., the candidate is 0.05 a.u. lower; centre the dedicated scan at 0.55 rather than applying a writeback from this experiment. If either fit is weak, its π marker is out of range or the first-read and joint π gains disagree relative to their errors, do not treat the difference as a gain correction. Repeat with more cycles for large shot SEM or extend the scan to bracket the first extremum if it is missing.
This experiment does not propose writeback. The model assumes independent thermalised cycles, Markov transitions between three states and three shared IQ centers with one isotropic noise width. A read that changes the state places its IQ between the initial and final centers. This does not identify multiple jumps within a read, cavity transients or a physical leakage level. Large Wald errors or negligible thermal population of a state can leave its dynamics poorly determined. Fit convergence or a small residual is not proof of unbiased parameters.

Layout reference: Ding, *Novel Gates with Superconducting Fluxonium Qubits*, Fig. 5-9(d) (printed p. 152), marks the chosen drive on a gain scan. That figure uses drive voltage and a designated state population; the first-read plot here uses DAC gain and projected IQ, and the population figure infers three anonymous states.

## Saved data and MCP recipe

Native Save retains the drive gain axis, per-point cycle count and 65 feature sums, and the 65 × 65 product sums pooled across points for the live joint projection. Load keeps the same data and can reproduce the analysis; the first-read shot SEM uses moments already present in these files.

`chain_amp_rabi` can set Drive pulse and Readout library references, the drive RF frequency and waveform duration, Drive gain › Start/Stop/Expts, Reps and Rounds. Use an override on `relax_delay` for Cycle thermalisation wait K. Omitted inputs keep the form values, overrides apply last, and null is passed to the form as an explicit clear where accepted. The recipe runs, analyzes and saves, with no writeback.
