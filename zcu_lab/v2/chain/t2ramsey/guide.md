**Readers:** operator, agent. Run the readout-chain T2 Ramsey from the GUI or MCP and read its live curves, joint fit and checks.

# K-cycle readout-chain T2 Ramsey

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Behavior

`chain/t2ramsey` sweeps the Ramsey wait like `twotone/t2ramsey`, but every repetition is one cycle of 16 reads instead of one read.
A cycle waits Cycle thermalisation wait K, then plays eight blocks of π/2 pulse, the swept Free-evolution wait, a second π/2 pulse and two reads of the same Readout, with no reset inside the cycle.
The second π/2 pulse has the first one's phase plus 180 degrees, so at zero delay it undoes the first and returns the qubit to its start.
Its phase also advances by 360 × Ramsey phase-ramp detune × delay degrees, which adds the specified fringes per µs of delay without changing the drive frequency.
Use at least 5 × T1 for K, including before the first cycle, to start near thermal equilibrium.
After a readout pulse ends, its `post_delay` sets the wait before the second readout or the next π/2 pulse.
For cavity ring-down, start with a wait of 5/(2πκ), with κ the readout linewidth in MHz and the result in µs. The fallback readout template leaves 0.015 µs of arriving drive on both sides of the ADC window.
The delay loop runs outside the cycle. Reps sets the cycles at every delay in one round, and Rounds sets the number of rounds.

The host reduces each cycle to 65 numbers: the IQ of the 16 reads, their I², IQ, Q² summed over three read groups, and IQ products between read pairs summed over six groups set by the number of Ramsey blocks and reads between them.
For a fixed sweep, the saved sums and the accumulator across rounds have fixed size, independent of the number of cycles.
Temporary arrays within one acquisition round grow with points × Reps. Increasing Reps therefore uses more temporary memory; increasing Rounds does not enlarge those per-round arrays.

## Parameter sources

π/2 pulse uses the `$qubit.module.x90` slot or a fresh `pi2_pulse` template, and Readout the `$resonator.module.readout` slot or a fresh `readout` template.
Select a single-tone pulse readout, not a mux module. The cycle has no Reset.
Free-evolution wait runs from 0 to 1.5 × `$qubit.t2r` over 101 points, and Ramsey phase-ramp detune starts at 5 / (1.5 × `$qubit.t2r`) MHz, which gives five fringes over the range. Both stay empty when T2r is missing.
Ramsey phase-ramp detune ramps the phase of the second π/2 pulse without changing its drive frequency.
Cycle thermalisation wait K starts at 5 * `$qubit.t1` µs, or 150 µs when T1 is missing.
Every cycle contains eight delays, so one cycle lasts about Cycle thermalisation wait K + 8 × (delay + 2 × π/2 length + 2 × readout length).
Reps starts at 100 and Rounds at 100. Reps must be at least 66, so that even a single delay gives the first round 65 cycles beyond its mean to estimate the noise of the 65 numbers.
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

Open this experiment with the qubit and its resonator bound in the selected point. Choose or expand the pulse and readout modules, check Free-evolution wait › Start, Stop and Expts, the number of sampled values, then Run. In `linrange`, enter endpoints and a point count. `center` uses a midpoint and span instead of endpoints. The `_step` variants use spacing instead of a point count. Missing inputs must be filled before Run.
Frequency is the RF carrier (MHz). Leave Mixer freq blank when no mixer setting is needed. Gen ch selects the drive output and NQZ its Nyquist zone. Gain, when shown, is the signed DAC full-scale fraction (a.u.). Phase sets the rotation axis in degrees. The waveform style and Length set the envelope. Pre-delay waits before playback and Post-delay waits after playback, both in µs.
For Readout, use the resonator's calibrated RF frequency, gain and phase. RO ch selects the receiver. RO length sets its integration window and Trig offset places that window, both in µs. An optional classifier stores state-calibration data but this sequence does not classify or filter shots.

## Run and Analyze

During Run, the upper live panel shows the mean IQ of the first read in each cycle. The I and Q weights follow the direction of greatest variation across the sweep; the projected signal is in a.u., not a classified state probability. With too few points to estimate that direction, the plot uses I. The lower panel combines the 65 features into a joint projection. It subtracts the across-sweep median and scales by noise estimated within each sweep point. Its unit is one point-mean standard error. It helps you see whether the joint data also resolve the fringes; there are no shot-SEM bars on this pooled, noise-normalized diagnostic. The header advances once per completed round and shows cycles accumulated per point. Stop retains completed rounds.

The first Analyze figure (`fit`) repeats the first-read live points (blue) and adds the seed model (orange). The vertical bars on both first-read plots are one **shot SEM**: the projected single-cycle sample standard deviation divided by the square root of the retained cycles at that point. Each cycle contributes its first read once. The bars come from the saved I, Q, I², IQ and Q² sums, not from round-to-round scatter or the uncertainty of a fitted parameter.
The solid damped cosine estimates the first-read fringe frequency and decay time. The dashed marker gives the first-read T2* within the scan. The fringe frequency is not the applied ramp.
In the first-read figure, R² is the fraction of variation explained by the fit; the normalized residual RMS divides residual RMS by the signal range. A negative R² means the curve fits worse than a constant at the mean. A smaller normalized residual RMS indicates closer agreement. Neither metric sets a selection threshold. The traces need enough contrast and sweep coverage to distinguish the fit from noise.

The `populations` figure tests whether a single physical model explains all 65 features across the sweep. Each panel shows one model-inferred first-read state population, not a classifier count. The points allow the operation to vary independently by sweep point. Their bars are one **Wald standard error**, taken from the local fit covariance, not shot SEM. The solid lines use the joint fit. The panel title gives the fitted thermal population before the drive. The plotted populations are after the drive. The y axes autoscale separately.
The joint Ramsey model uses exp(−wait/T2*) × cos(2π × fringe frequency × wait) and a separate decay for the third-state row. T2* includes relaxation and dephasing; the dashed line marks it within the scan. The fitted fringe frequency includes the programmed phase ramp and the drive–qubit mismatch. The cosine does not measure its sign.
State numbers are fit labels, not independently identified levels. R is 1/(2π × sweep span × fringe-frequency standard error), using the Ramsey fringe frequency in inverse sweep units. It is a frequency-precision diagnostic, not a pass/fail test. The joint weighted residual J is reported with its degrees of freedom (df). The separate checks report their own J/df and give p where available:

- **within-point** asks whether a free three-state model explains the data at each point.
- **sharing** checks whether sharing readout parameters worsens the fit relative to fitting points separately. It has no p-value.
- **curve** checks the constrained sweep model against a free operation at each point.

Compare the points and the model before interpreting the numbers. Large J relative to df or a small reported p indicates a mismatch to investigate; no check selects a calibration automatically. Missing diagnostics are not a pass. The Analyze summary identifies the run and gives its saved Reps×Rounds setting, not the number of rounds that finished.

## What to do next

Take at least 10 points per fringe period (step ≤ 1/(10 × fringe frequency)) and a window near 3 × T2*. For a 0.4 MHz fringe and T2* ≈ 12 µs, use steps ≤ 0.25 µs and a stop near 36 µs, at least 145 points from zero. Sparse sampling can create a slower apparent fringe; a short window leaves the envelope set mostly by extrapolation. Do not infer a signed qubit-frequency correction from this fit alone. Use a second run with a known different ramp to resolve the sign or the dedicated `twotone/t2ramsey` calibration. The summary does not retain a separate measured ramp.
This experiment does not propose writeback. The model assumes independent thermalised cycles, Markov transitions between three states and three shared IQ centers with one isotropic noise width. A read that changes the state places its IQ between the initial and final centers. This does not identify multiple jumps within a read, cavity transients or a physical leakage level. Large Wald errors or negligible thermal population of a state can leave its dynamics poorly determined. Fit convergence or a small residual is not proof of unbiased parameters.

Layout reference: Ding, *Novel Gates with Superconducting Fluxonium Qubits*, Fig. 5-9(b,e) (printed p. 152), places a phase-sensitive sequence next to its fringes. This run varies free-evolution wait, with a phase ramp on the second π/2 pulse, and measures projected IQ or model-inferred populations rather than a calibrated phase.

## Saved data and MCP recipe

Native Save retains the free-evolution wait axis, per-point cycle count and 65 feature sums, and the 65 × 65 product sums pooled across points for the live joint projection. Load keeps the same data and can reproduce the analysis; the first-read shot SEM uses moments already present in these files.

`chain_t2ramsey` can set π/2 pulse and Readout library references, Free-evolution wait › Stop/Expts, Ramsey phase-ramp detune, Reps and Rounds. Use an override on `relax_delay` for Cycle thermalisation wait K. Omitted inputs keep the form values, overrides apply last, and null is passed to the form as an explicit clear where accepted. The recipe runs, analyzes and saves, with no writeback.
