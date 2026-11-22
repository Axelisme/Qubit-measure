# K-cycle readout-chain amplitude Rabi

**Readers:** operator, agent. Run the readout-chain gain Rabi from the GUI or MCP and read its live curves, joint fit and residuals.

**Last updated:** 2026-11-22

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Behavior

`chain/amp_rabi` sweeps the Rabi Pulse gain like `twotone/rabi/amp_rabi`, but every repetition is one cycle of 16 reads instead of one read.
A cycle waits Relax delay, then plays eight blocks of the gain-swept Rabi Pulse followed by two reads of the same Readout, with no reset inside the cycle.
Relax delay is the thermalisation wait K, including before the first cycle. Set it to at least 5 T1 for the model's thermal-equilibrium start.
After a readout pulse ends, its `post_delay` sets the wait before the second readout or the next Rabi Pulse.
The next pulse keeps its own pre-delay but may start before the previous ADC window closes. Set the timing so windows on the same ADC do not overlap.
The gain loop runs outside the cycle. Reps sets the cycles at every gain in one round, and Rounds sets the number of rounds.

The host reduces each cycle to 65 numbers: the IQ of the 16 reads, their I², IQ, Q² summed over three read groups, and IQ products between read pairs summed over six groups set by the number of Rabi pulses and reads between them.
For a fixed sweep, the saved sums and the accumulator across rounds have fixed size, independent of the number of cycles.
Temporary arrays within one acquisition round grow with points × Reps. Increasing Reps therefore uses more temporary memory; increasing Rounds does not enlarge those per-round arrays.

## Parameter sources

The Form uses the `twotone/rabi/amp_rabi` sources: the x180 copy or a fresh `qub_probe` pulse, the `$resonator.module.readout` slot or a fresh `readout` template, and 51 gains from -0.3 to `min(1.0, 1.2 * x180 gain)`.
Readout must be a pulse readout; a mux readout module is rejected by the Form. The cycle has no Reset or Init Pulse.
Relax delay starts at 5 * `$qubit.t1` µs, or 150 µs when T1 is missing.
Reps starts at 100 and Rounds at 100. Reps must be at least 66, so that even a single gain gives the first round 65 cycles beyond its mean to estimate the noise of the 65 numbers.
One cycle lasts about Relax delay + 8 × (pulse length + 2 × readout length).
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

## Live plot

After every round the `measurement` plot shows two curves from all completed rounds, both in standard errors of the point means (sigma):

- `all 65 numbers`: the 65 numbers relative to their median over gains, whitened by the noise within a gain and projected onto the direction that changes most across the sweep.
- `first read`: the mean IQ of read 0 relative to its median, projected onto its principal axis; this is the familiar single-read Rabi curve.

The sign of the first curve follows the second, so both oscillate together. Stop keeps the sums of completed rounds. A stop before the first round finishes keeps nothing and the Run fails with that message.

## Analysis

Analysis needs no options. It fits a cosine to the mean IQ projection of the first read. The seed π gain x_π gives the start f = 1 / (2 x_π).
It then fits all 65 numbers at all gains jointly with a three-state model whose readout and population parameters are shared across gains.
The drive row for each starting state interpolates between two free population endpoints, D_i(x) = [1 − f_i(x)] × endpoint_i0 + f_i(x) × endpoint_i1.
The curve weights are sin²(π f x) for g, cos²(π f x) for e and sin²(π f x) for the third state.
The third-state row can vary with x. Equal endpoints give a flat row; distinct endpoints allow an oscillating response.
f is in cycles per gain unit, and the π gain is 1 / (2 f).

In `joint residuals`, the upper panel shows the measured points from the live `all 65 numbers` curve and a solid joint-model curve on the same projection. A dashed line marks the joint π gain when it falls inside the sweep.
The lower panel subtracts the joint prediction from the measurement at each acquired gain. Look there for gain ranges where the model consistently over- or underpredicts the projected signal.
Both panels use standard errors of point means estimated from the pooled noise. They do not show populations or per-gain shot error bars.

The header and summary report the joint π gain and its Wald standard error, R and the following residual numbers:

- `J` sums the squares of the whitened residuals from the last joint fit stage. `df` is 65 times the number of gains minus the number of joint fit parameters.
- `j_per_df` is J/df. `expected_band_low` and `expected_band_high` are the endpoints of the approximate χ² reference band 1 ± 2√(2/df).
- `tail_fraction` is the fraction of all whitened components with |r| > 3. The header shows it as a percentage. These are the 65-component residuals used by the joint fit, not the one-dimensional residuals in the lower panel.

The summary also reports the fitted frequency and its Wald error, and the first-read π gain. R = 1 / (2π × gain span × frequency standard error) measures how the frequency error compares with the finite gain span. The analysis uses no cutoff on these numbers.
The `fit` figure shows the cosine used to seed the joint fit. Its dashed lines mark the seed π and π/2 gains, and the shaded bands show their parameter standard errors. Its R² compares the seed fit with the first-read data only.
If the seed or joint fit fails, the live plot and saved sums remain. There is no secondary diagnostic fit.

This experiment proposes no writeback. Compare `pi_amp` with `first_read_pi_amp`, and calibrate x180 with `twotone/rabi/amp_rabi`.

### Model assumptions and starting fit

The fit assumes independent thermalised cycles and Markov state transitions within each cycle.
It shares the three IQ centers, one isotropic Gaussian noise variance, the thermal populations and the read-transition matrix across sweep points.
Gaussian noise is independent between reads and cycles. The model does not fit a separate noise width or ellipse for each state.

The read matrix is R = diag(r) + outer(1 − r, landing), with three retention values r and one landing population vector.
For a transition i → j, the conditional IQ center lies on the line between the two state centers, at c_j + u × (c_i − c_j).
The model draws u from a truncated exponential on [0, 1] with rate −log(1 − R[i,j]). A read without a transition uses c_i.
This emission law is not a general model of multiple jumps during integration or resonator dynamics.
Transitions during a long gap after integration need not produce the IQ tail that this model assigns to a read transition.
A small residual does not establish unbiased physical parameters. The reported Wald errors measure local fit uncertainty. Low state contrast or few cycles can make them understate uncertainty.

The first-read mean must support the starting fit. Weak or absent first-read contrast can make that fit fail or supply an uninformative seed even when later reads have correlations.
The joint fit starts from the first-read curve parameters. It does not try multiple curve seeds or guarantee the correct basin.
There is no manual initial value or fallback.

## Saved data

The saved record contains the gain axis, the cycle count and the 65 sums at each gain, and one 65 × 65 product sum added over all gains.
The product sum sets the pooled noise scale for the live curve and the Analyze projection. It does not set the joint fit weights. Load the file and select Analyze to rerun the fit from the saved sums.

## MCP recipe

`chain_amp_rabi` takes `gain_range`, `points`, `reps`, `rounds`, `frequency_mhz`, `pulse_length_us`, `drive_ref` and `readout_ref`; points maps to gain Expts.
Set Relax delay with an override on `relax_delay`.
The recipe runs, analyzes and saves. It never asks for writeback.
The report gives `pi_gain`, `pi_gain_err`, `frequency_per_gain`, `frequency_per_gain_err`, `reliability`, `first_read_pi_gain` and the six residual fields listed above. The projected residuals at each gain stay in the figure.
Omitted inputs retain Form values; overrides apply last, then check_ready reports missing or invalid fields before Run.
