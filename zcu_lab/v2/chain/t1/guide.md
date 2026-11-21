**Readers:** operator, agent. Run the readout-chain T1 from the GUI or MCP and read its live curves, joint fit and checks.

# K-cycle readout-chain T1

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Behavior

`chain/t1` sweeps the wait after a π pulse like `twotone/t1`, but every repetition is one cycle of 16 reads instead of one read.
A cycle waits Cycle thermalisation wait K, then plays eight blocks of the π pulse, the swept Wait after π pulse and two reads of the same Readout, with no reset inside the cycle.
Use at least 5 × T1 for K, including before the first cycle, to start near thermal equilibrium.
After a readout pulse ends, its `post_delay` sets the wait before the second readout or the next π pulse.
For cavity ring-down, start with a wait of 5/(2πκ), with κ the readout linewidth in MHz and the result in µs. The fallback readout template leaves 0.015 µs of arriving drive on both sides of the ADC window.
The delay loop runs outside the cycle. Reps sets the cycles at every delay in one round, and Rounds sets the number of rounds.

The host reduces each cycle to 65 numbers: the IQ of the 16 reads, their I², IQ, Q² summed over three read groups, and IQ products between read pairs summed over six groups set by the number of π pulses and reads between them.
For a fixed sweep, the saved sums and the accumulator across rounds have fixed size, independent of the number of cycles.
Temporary arrays within one acquisition round grow with points × Reps. Increasing Reps therefore uses more temporary memory; increasing Rounds does not enlarge those per-round arrays.

## Parameter sources

The Form uses the `twotone/t1` sources: the `$qubit.module.x180` slot or a fresh `pi_pulse` template, and the `$resonator.module.readout` slot or a fresh `readout` template.
Select a single-tone pulse readout, not a mux module. The cycle has no Reset.
Cycle thermalisation wait K starts at 5 * `$qubit.t1` µs, or 150 µs when T1 is missing. Wait after π pulse runs from 0 to the same value over 101 points.
Every cycle contains eight delays, so one cycle lasts about Cycle thermalisation wait K + 8 × (delay + π length + 2 × readout length). A shorter delay range shortens long-delay cycles.
Reps starts at 100 and Rounds at 100. Reps must be at least 66, so that even a single delay gives the first round 65 cycles beyond its mean to estimate the noise of the 65 numbers.
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

Open this experiment with the qubit and its resonator bound in the selected point. Choose or expand the pulse and readout modules, check Wait after π pulse › Start, Stop and Expts, the number of sampled values, then Run. In `linrange`, enter endpoints and a point count. `center` uses a midpoint and span instead of endpoints. The `_step` variants use spacing instead of a point count. Missing inputs must be filled before Run.
Frequency is the RF carrier (MHz). Leave Mixer freq blank when no mixer setting is needed. Gen ch selects the drive output and NQZ its Nyquist zone. Gain, when shown, is the signed DAC full-scale fraction (a.u.). Phase sets the rotation axis in degrees. The waveform style and Length set the envelope. Pre-delay waits before playback and Post-delay waits after playback, both in µs.
For Readout, use the resonator's calibrated RF frequency, gain and phase. RO ch selects the receiver. RO length sets its integration window and Trig offset places that window, both in µs. An optional classifier stores state-calibration data but this sequence does not classify or filter shots.

## Run and Analyze

During Run, the upper live panel shows the mean IQ of the first read in each cycle. The I and Q weights follow the direction of greatest variation across the sweep; the projected signal is in a.u., not a classified state probability. With too few points to estimate that direction, the plot uses I. The lower panel combines the 65 features into a joint projection. It subtracts the across-sweep median and scales by noise estimated within each sweep point. Its unit is one point-mean standard error. It helps you see whether the joint data also resolve the decay; there are no shot-SEM bars on this pooled, noise-normalized diagnostic. The header advances once per completed round and shows cycles accumulated per point. Stop retains completed rounds.

The first Analyze figure (`fit`) repeats the first-read live points (blue) and adds the seed model (orange). The vertical bars on both first-read plots are one **shot SEM**: the projected single-cycle sample standard deviation divided by the square root of the retained cycles at that point. Each cycle contributes its first read once. The bars come from the saved I, Q, I², IQ and Q² sums, not from round-to-round scatter or the uncertainty of a fitted parameter.
The solid single-exponential line tests whether the first read resolves a decay. Its dashed time marker is drawn inside the scan only. This first-read time starts the three-state rate-equation fit and is not the reported T1.
In the first-read figure, R² is the fraction of variation explained by the fit; the normalized residual RMS divides residual RMS by the signal range. A negative R² means the curve fits worse than a constant at the mean. A smaller normalized residual RMS indicates closer agreement. Neither metric sets a selection threshold. The traces need enough contrast and sweep coverage to distinguish the fit from noise.

The `populations` figure tests whether a single physical model explains all 65 features across the sweep. Each panel shows one model-inferred first-read state population, not a classifier count. The points allow the operation to vary independently by sweep point. Their bars are one **Wald standard error**, taken from the local fit covariance, not shot SEM. The solid lines use the joint fit. The panel title gives the fitted thermal population before the drive. The plotted populations are after the drive. The y axes autoscale separately.
The rate model relaxes after the π pulse toward the thermal populations. T1 is the eigen time carrying the larger amplitude of population change, not necessarily the longer time. Secondary time is the other eigen time. The dashed line marks T1 if it is within the sweep. The names g, e and f rank the fitted thermal populations from high to low; that ranking is not an independent identification of the levels. The bottom residual panel plots (point − model) / Wald SE in the matching state colours. Its grey band is ±1.96, a normal-reference range, not a selection gate. Persistent residuals with one sign indicate a mismatch.
The joint weighted residual J is reported with its degrees of freedom (df). The separate checks report their own J/df and give p where available:

- **within-point** asks whether a free three-state model explains the data at each point.
- **sharing** checks whether sharing readout parameters worsens the fit relative to fitting points separately. It has no p-value.
- **curve** checks the constrained sweep model against a free operation at each point.

Compare the points and the model before interpreting the numbers. Large J relative to df or a small reported p indicates a mismatch to investigate; no check selects a calibration automatically. Missing diagnostics are not a pass. The Analyze summary identifies the run and gives its saved Reps×Rounds setting, not the number of rounds that finished.

## What to do next

If the data show only the beginning of the decay, set Wait after π pulse › Stop near 3 × the tentative T1 and retain points near zero. For example, for a 20 µs candidate use about 60 µs. Check the repeat against both decay modes before using a dedicated `twotone/t1` run to update the qubit calibration. If timing or amplitudes of the modes are close, their order can switch across runs; the reported errors do not account for that choice.
This experiment does not propose writeback. The model assumes independent thermalised cycles, Markov transitions between three states and three shared IQ centers with one isotropic noise width. A read that changes the state places its IQ between the initial and final centers. This does not identify multiple jumps within a read, cavity transients or a physical leakage level. Large Wald errors or negligible thermal population of a state can leave its dynamics poorly determined. Fit convergence or a small residual is not proof of unbiased parameters.

Layout reference: Ding, *Novel Gates with Superconducting Fluxonium Qubits*, Fig. 5-13(b) (printed p. 157), separates measured points from model lines; its benchmarking survival and sequence-length axis differ from the readout chain's population and post-π wait.

## Saved data and MCP recipe

Native Save retains the wait after π pulse axis, per-point cycle count and 65 feature sums, and the 65 × 65 product sums pooled across points for the live joint projection. Load keeps the same data and can reproduce the analysis; the first-read shot SEM uses moments already present in these files.

`chain_t1` can set π pulse and Readout library references, Wait after π pulse › Stop/Expts, Reps and Rounds. Use an override on `relax_delay` for Cycle thermalisation wait K. Omitted inputs keep the form values, overrides apply last, and null is passed to the form as an explicit clear where accepted. The recipe runs, analyzes and saves, with no writeback.
