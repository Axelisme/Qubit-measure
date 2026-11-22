**Readers:** operator, agent. Run the readout-chain T2 Hahn echo from the GUI or MCP and read its live curves, joint fit and residuals.

# K-cycle readout-chain T2 echo

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Behavior

`chain/t2echo` sweeps the total echo wait like `twotone/t2echo`, but every repetition is one cycle of 16 reads instead of one read.
A cycle waits Cycle thermalisation wait K, then plays eight blocks of π/2 pulse, half the Delay, π pulse, the other half, π/2 pulse again and two reads of the same Readout, with no reset inside the cycle.
With the π pulse on the π/2 pulse's axis, a zero-delay block is a full 2π rotation and returns the qubit to its start. There is no phase ramp.
Use at least 5 × T1 for K, including before the first cycle, to start near thermal equilibrium.
After a readout pulse ends, its `post_delay` sets the wait before the second readout or the next π/2 pulse.
For cavity ring-down, start with a wait of 5/(2πκ), with κ the readout linewidth in MHz and the result in µs. The fallback readout template leaves 0.015 µs of arriving drive on both sides of the ADC window.
The delay loop runs outside the cycle. Delay is the total wait; both halves are rounded to the hardware time grid. Reps sets the cycles at every delay in one round, and Rounds sets the number of rounds.

The host reduces each cycle to 65 numbers: the IQ of the 16 reads, their I², IQ, Q² summed over three read groups, and IQ products between read pairs summed over six groups set by the number of echo blocks and reads between them.
For a fixed sweep, the saved sums and the accumulator across rounds have fixed size, independent of the number of cycles.
Temporary arrays within one acquisition round grow with points × Reps. Increasing Reps therefore uses more temporary memory; increasing Rounds does not enlarge those per-round arrays.

## Parameter sources

The Form uses the `twotone/t2echo` sources: the `$qubit.module.x90` slot or a fresh `pi2_pulse` template, the `$qubit.module.x180` slot or a fresh `pi_pulse` template, and the `$resonator.module.readout` slot or a fresh `readout` template.
Select a single-tone pulse readout, not a mux module. The cycle has no Reset.
Total echo wait runs from 0 to 1.5 × `$qubit.t2e` over 101 points; it stays empty when T2e is missing.
Cycle thermalisation wait K starts at 5 * `$qubit.t1` µs, or 150 µs when T1 is missing.
Every cycle contains eight echoes, so one cycle lasts about Cycle thermalisation wait K + 8 × (delay + π length + 2 × π/2 length + 2 × readout length).
Reps starts at 100 and Rounds at 100. Reps must be at least 66, so that even a single delay gives the first round 65 cycles beyond its mean to estimate the noise of the 65 numbers.
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

Open this experiment with the qubit and its resonator bound in the selected point. Choose or expand the pulse and readout modules, check Total echo wait › Start, Stop and Expts, the number of sampled values, then Run. In `linrange`, enter endpoints and a point count. `center` uses a midpoint and span instead of endpoints. The `_step` variants use spacing instead of a point count. Missing inputs must be filled before Run.
Frequency is the RF carrier (MHz). Leave Mixer freq blank when no mixer setting is needed. Gen ch selects the drive output and NQZ its Nyquist zone. Gain, when shown, is the signed DAC full-scale fraction (a.u.). Phase sets the rotation axis in degrees. The waveform style and Length set the envelope. Pre-delay waits before playback and Post-delay waits after playback, both in µs.
For Readout, use the resonator's calibrated RF frequency, gain and phase. RO ch selects the receiver. RO length sets its integration window and Trig offset places that window, both in µs. An optional classifier stores state-calibration data but this sequence does not classify or filter shots.

## Run and Analyze

During Run, the upper live panel shows the mean IQ of the first read in each cycle. The I and Q weights follow the direction of greatest variation across the sweep; the projected signal is in a.u., not a classified state probability. With too few points to estimate that direction, the plot uses I. The lower panel combines the 65 features into a joint projection. It subtracts the across-sweep median and scales by noise estimated within each sweep point. Its unit is one point-mean standard error. It helps you see whether the joint data also resolve the decay; there are no shot-SEM bars on this pooled, noise-normalized diagnostic. The header advances once per completed round and shows cycles accumulated per point. Stop retains completed rounds.

The first Analyze figure (`fit`) repeats the first-read live points. The vertical bars on both first-read plots are one **shot SEM**: the projected single-cycle sample standard deviation divided by the square root of the retained cycles at that point. Each cycle contributes its first read once. The bars come from the saved I, Q, I², IQ and Q² sums, not from round-to-round scatter or the uncertainty of a fitted parameter.
The first-read exponential fit excludes the first sweep point, shown in grey. Run and Analyze use the remaining points to set the IQ projection direction. When the first-read fit supplies an estimate, the orange curve shows the model and a dashed marker shows its echo time if it lies in the scan. That time starts both joint decay times.
If fewer than two distinct waits remain after excluding the first point, Analyze skips the first-read fit. If the fit is skipped or cannot estimate a decay time, the figure connects the measured points and shot SEM and shows “First-read fit unavailable” without a curve, marker or fit statistics. Analyze still attempts the joint fit once using all saved waits. It starts the echo time at one third of the full saved wait span and the third-state time at the full span. Those starts are not estimates. The summary reports `first_read_t2` as null if the joint fit succeeds without a first-read estimate.

When the first-read fit is available, R² is the fraction of variation explained by the fit; the normalized residual RMS divides residual RMS by the signal range. A negative R² means the curve fits worse than a constant at the mean. A smaller normalized residual RMS indicates closer agreement. Neither metric sets a selection threshold. The traces need enough contrast and sweep coverage to distinguish the fit from noise.

The upper panel of `joint-residuals` shows the measured points and solid joint model curve for the same all-65-feature projection as the lower live panel. It is a projected signal, not an inferred state population. The lower panel shows measured minus model at each total echo wait in the same SEM units. Look for residual patterns such as a drift or fringe that the model curve does not explain. The pooled noise scale does not give a shot-SEM bar for each delay.

Analyze lists the joint figure first. If the saved statistics cannot support the live projection, the figure says “Live-direction projection unavailable” and shows the joint estimates and residual diagnostics without curves. The first-read figure remains available.

The figure lists T2 echo and the third-state decay time with their Wald standard errors in µs. The joint model uses total echo wait in its exponential envelope. The most populated thermal row identifies the qubit branch. If the model's third row is more populated than the qubit rows, the two fitted times exchange roles in the report. The dashed line marks the reported T2 echo when it lies in the sweep.

The scalar diagnostics use all 65 whitened residual components at every delay, not just the projection shown in the panels. `J` is their sum of squares. `df` is 65 times the number of delays minus the number of fitted parameters. The figure and summary give `J/df` and the numeric endpoints of the approximate χ² reference band `1 ± 2√(2/df)`. The tail fraction is the share of whitened components with `|r| > 3`. The band and tail fraction help put the residuals in context. Neither is an acceptance threshold or proof of accurate decay times. The Analyze summary identifies the run and gives its saved Reps×Rounds setting, not the number of rounds that finished.

## What to do next

Set Total echo wait › Stop near 3 × the tentative decay time, for example about 45 µs for a 15 µs estimate. A short window leaves only a shallow part of the decay so the time depends on extrapolation. Echo fits no intentional fringe. If residual fringes are visible, sample at least 10 points per period to resolve them and check pulse calibration rather than interpreting aliased ripple as decay. For an observed 2 µs period use steps no larger than 0.2 µs. Sparse points can hide or alias a ripple. Use `twotone/t2echo` for a separate calibration.
This experiment does not propose writeback. The model assumes independent thermalised cycles, Markov transitions between three states and three shared IQ centers with one isotropic noise width. A read that changes the state places its IQ between the initial and final centers. This does not identify multiple jumps within a read, cavity transients or a physical leakage level. Large Wald errors or negligible thermal population of a state can leave its dynamics poorly determined. Fit convergence or a small residual is not proof of unbiased parameters.

Layout reference: Ding, *Novel Gates with Superconducting Fluxonium Qubits*, Fig. 5-9(b,e) (printed p. 152), uses pulse-sequence context to read a phase trace. Here the echo refocuses phase and the decay is measured against total wait, not an applied phase axis.

## Saved data and MCP recipe

Native Save retains the total echo wait axis, per-point cycle count and 65 feature sums, and the 65 × 65 product sums pooled across points for the live joint projection. Load keeps the same data and can reproduce the analysis; the first-read shot SEM uses moments already present in these files.

`chain_t2echo` can set π/2 pulse, π pulse and Readout library references, Total echo wait › Stop/Expts, Reps and Rounds. Use an override on `relax_delay` for Cycle thermalisation wait K. Omitted inputs keep the form values, overrides apply last, and null is passed to the form as an explicit clear where accepted. The recipe runs, analyzes and saves, with no writeback. Its report includes the decay times, their standard errors, `first_read_t2_us` (null without a first-read estimate), and the same scalar joint residual diagnostics as Analyze. The per-delay projection stays in the figure.
