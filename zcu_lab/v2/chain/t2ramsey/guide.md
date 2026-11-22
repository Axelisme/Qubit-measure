# K-cycle readout-chain T2 Ramsey

**Readers:** operator, agent. Run the readout-chain T2 Ramsey from the GUI or MCP and read its live curves, joint fit and checks.

**Last updated:** 2026-11-09

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Behavior

`chain/t2ramsey` sweeps the Ramsey wait like `twotone/t2ramsey`, but every repetition is one cycle of 16 reads instead of one read.
A cycle waits Relax delay, then plays eight blocks of Pi/2 Pulse, the swept Delay, a second Pi/2 Pulse and two reads of the same Readout, with no reset inside the cycle.
The second Pi/2 Pulse has the first one's phase plus 180 degrees, so at zero delay it undoes the first and returns the qubit to its start.
Its phase also advances by 360 × Detune × delay degrees, which adds Detune fringes per µs of delay without changing the drive frequency.
Relax delay is the thermalisation wait K, including before the first cycle. Set it to at least 5 T1 for the model's thermal-equilibrium start.
After a readout pulse ends, its `post_delay` sets the wait before the second readout or the next Pi/2 Pulse.
The next pulse keeps its own pre-delay but may start before the previous ADC window closes. Set the timing so windows on the same ADC do not overlap.
The delay loop runs outside the cycle. Reps sets the cycles at every delay in one round, and Rounds sets the number of rounds.

The host reduces each cycle to 65 numbers: the IQ of the 16 reads, their I², IQ, Q² summed over three read groups, and IQ products between read pairs summed over six groups set by the number of Ramsey blocks and reads between them.
For a fixed sweep, the saved sums and the accumulator across rounds have fixed size, independent of the number of cycles.
Temporary arrays within one acquisition round grow with points × Reps. Increasing Reps therefore uses more temporary memory; increasing Rounds does not enlarge those per-round arrays.

## Parameter sources

Pi/2 Pulse uses the `$qubit.module.x90` slot or a fresh `pi2_pulse` template, and Readout the `$resonator.module.readout` slot or a fresh `readout` template.
Readout must be a pulse readout; a mux readout module is rejected by the Form. The cycle has no Reset.
Delay runs from 0 to 1.5 × `$qubit.t2r` over 101 points, and Detune starts at 5 / (1.5 × `$qubit.t2r`) MHz, which gives five fringes over the range. Both stay empty when T2r is missing.
Detune ramps the phase of the second π/2 pulse without changing its drive frequency.
Relax delay starts at 5 * `$qubit.t1` µs, or 150 µs when T1 is missing.
Every cycle contains eight delays, so one cycle lasts about Relax delay + 8 × (delay + 2 × π/2 length + 2 × readout length).
Reps starts at 100 and Rounds at 100. Reps must be at least 66, so that even a single delay gives the first round 65 cycles beyond its mean to estimate the noise of the 65 numbers.
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

## Live plot

After every round the `measurement` plot shows two curves from all completed rounds, both in standard errors of the point means (sigma):

- `all 65 numbers`: the 65 numbers relative to their median over delays, whitened by the noise within a delay and projected onto the direction that changes most across the sweep.
- `first read`: the mean IQ of read 0 relative to its median, projected onto its principal axis; this is the familiar single-read Ramsey curve.

The sign of the first curve follows the second, so both show the same fringes. Stop keeps the sums of completed rounds. A stop before the first round finishes keeps nothing and the Run fails with that message.

## Analysis

Analysis needs no options. It first fits the first read's mean IQ with the `twotone/t2ramsey` decaying-fringe fit and publishes that experiment's `fit` figure; its T2 starts both fitted times and its fringe frequency starts the detuning.
It then fits all 65 numbers at all delays jointly with `t2_ramsey_model`, a three-state model whose readout and population nuisances are shared across delays.
With c = exp(−x / t2) × cos(2π × detuning × x), the probability of ending in e is (1 − c)/2 from g and (1 + c)/2 from e; the third state decays with its own exp(−x / tf).
t2 includes the transverse part of T1 relaxation. detuning is the fringe frequency in MHz and includes the Detune ramp; its sign is not measured.

The `populations` plot shows, per delay, the first-read populations of a fit with a free operation at each delay, with Wald error bars, and the fitted model as lines.
State labels are arbitrary; the legend gives each state's fitted thermal population, and the largest one is normally g.
The title gives t2, detuning, tf, R and the three lack-of-fit checks as J/df:

- `within-point`: the free three-state model at each delay, with p.
- `sharing`: shared readout nuisances against fully separate delays; it has no p.
- `curve`: the Ramsey curves against a free operation at each delay, with p. Pulse errors or other third-state dynamics raise it.

The summary reports `t2`, `detuning`, `tf` and their `_err` values, the reliability `R` = 1 / (2π × delay span × detuning_err), the three checks, the joint `J` and `df`, and `first_read_t2` and `first_read_detuning` from the first-read fit.
Errors are Wald errors from the joint fit. At low state contrast and few cycles per delay they can understate the uncertainty. A small third-state population leaves tf poorly determined. R is a diagnostic without a threshold.
A fit failure in either step produces an analysis failure; the live plot and the saved sums remain.

This experiment proposes no writeback. Compare `t2` with `first_read_t2`, and calibrate `$qubit.t2r` and the qubit frequency with `twotone/t2ramsey`.

### Model assumptions and starting fit

The fit assumes independent thermalised cycles and Markov state transitions within each cycle.
It shares the three IQ centers, one isotropic Gaussian noise variance, the thermal populations and the read-transition matrix across sweep points.
Gaussian noise is independent between reads and cycles. The model does not fit a separate noise width or ellipse for each state.

The read matrix is R = diag(r) + outer(1 − r, landing), with three retention values r and one landing population vector.
For a transition i → j, the conditional IQ center lies on the line between the two state centers, at c_j + u × (c_i − c_j).
The model draws u from a truncated exponential on [0, 1] with rate −log(1 − R[i,j]). A read without a transition uses c_i.
This emission law is not a general model of multiple jumps during integration or resonator dynamics.
Transitions during a long gap after integration need not produce the IQ tail that this model assigns to a read transition.
Passing the lack-of-fit checks does not establish unbiased physical parameters.

The first-read mean must support the starting fit. Weak or absent first-read contrast can make that fit fail or supply an uninformative seed even when later reads have correlations.
The joint fit starts from the first-read curve parameters. It does not try multiple curve seeds or guarantee the correct basin.
There is no manual initial value or fallback.

## Saved data

The saved record contains the delay axis, the cycle count and the 65 sums at each delay, and one 65 × 65 product sum added over all delays.
Delays are us in memory and seconds in the file. The product sum feeds only the live plot. Loading the file and analyzing it again gives the same summary.

## MCP recipe

`chain_t2ramsey` takes `max_delay_us`, `points`, `detune_mhz`, `reps`, `rounds`, `pi2_ref` and `readout_ref`; points maps to delay Expts.
Set Relax delay with an override on `relax_delay`.
The recipe runs, analyzes and saves. It never asks for writeback.
The report gives `t2_us`, `t2_err_us`, `detuning_mhz`, `detuning_err_mhz`, `tf_us`, `tf_err_us`, `reliability`, the checks `within_point`, `sharing` and `curve` (each J, df, p), and `first_read_t2_us`.
Omitted inputs retain Form values; overrides apply last, then check_ready reports missing or invalid fields before Run.
