# K-cycle readout-chain T1

**Readers:** operator, agent. Run the readout-chain T1 from the GUI or MCP and read its live curves, joint fit and checks.

**Last updated:** 2026-11-09

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Behavior

`chain/t1` sweeps the wait after a π pulse like `twotone/t1`, but every repetition is one cycle of 16 reads instead of one read.
A cycle waits Relax delay, then plays eight blocks of the Pi Pulse, the swept Delay and two reads of the same Readout, with no reset inside the cycle.
Relax delay is the thermalisation wait K, including before the first cycle. Set it to at least 5 T1 for the model's thermal-equilibrium start.
After a readout pulse ends, its `post_delay` sets the wait before the second readout or the next Pi Pulse.
The next pulse keeps its own pre-delay but may start before the previous ADC window closes. Set the timing so windows on the same ADC do not overlap.
The delay loop runs outside the cycle. Reps sets the cycles at every delay in one round, and Rounds sets the number of rounds.

The host reduces each cycle to 65 numbers: the IQ of the 16 reads, their I², IQ, Q² summed over three read groups, and IQ products between read pairs summed over six groups set by the number of π pulses and reads between them.
For a fixed sweep, the saved sums and the accumulator across rounds have fixed size, independent of the number of cycles.
Temporary arrays within one acquisition round grow with points × Reps. Increasing Reps therefore uses more temporary memory; increasing Rounds does not enlarge those per-round arrays.

## Parameter sources

The Form uses the `twotone/t1` sources: the `$qubit.module.x180` slot or a fresh `pi_pulse` template, and the `$resonator.module.readout` slot or a fresh `readout` template.
Readout must be a pulse readout; a mux readout module is rejected by the Form. The cycle has no Reset.
Relax delay starts at 5 * `$qubit.t1` us, or 150 us when T1 is missing. Delay runs from 0 to the same value over 101 points.
Every cycle contains eight delays, so one cycle lasts about Relax delay + 8 × (delay + π length + 2 × readout length). A shorter delay range shortens long-delay cycles.
Reps starts at 100 and Rounds at 100. Reps must be at least 66, so that even a single delay gives the first round 65 cycles beyond its mean to estimate the noise of the 65 numbers.
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

## Live plot

After every round the `measurement` plot shows two curves from all completed rounds, both in standard errors of the point means (sigma):

- `all 65 numbers`: the 65 numbers relative to their median over delays, whitened by the noise within a delay and projected onto the direction that changes most across the sweep.
- `first read`: the mean IQ of read 0 relative to its median, projected onto its principal axis; this is the familiar single-read T1 curve.

The sign of the first curve follows the second, so both decay together. Stop keeps the sums of completed rounds. A stop before the first round finishes keeps nothing and the Run fails with that message.

## Analysis

Analysis needs no options. It first fits the first read's mean IQ with a single decay. That T1 only starts the joint fit and has no figure of its own.
It then fits all 65 numbers at all delays jointly with a three-state rate-equation model whose readout and population nuisances are shared across delays.

The operation at delay x is D(x) = D₀ exp(G x). D₀ is a free stochastic matrix: what the π pulse does at zero delay.
G holds three non-negative pair rates in detailed balance with the fitted thermal populations, so the populations relax towards thermal equilibrium.

G has two relaxation modes, each with an eigen time and an amplitude. The amplitude is the population change the mode carries from the state after D₀, summed over starting states with their thermal weights.

- `T1` is the eigen time with the larger amplitude. It is the time scale that carries most of the population change after the π pulse, and it can be the shorter of the two times.
- The `secondary time` is the other eigen time.

When the two amplitudes or the two times are close, which one is called T1 can change between runs, and the Wald errors do not include that choice. A mode with a zero decay rate has an infinite time and error, which the summary reports as null.

The `populations` figure is the only analysis figure. Its upper panel shows, per delay, the first-read populations of a fit with a free operation at each delay, with Wald error bars, and the fitted model as lines.
States are named by their fitted thermal population: g is the most populated, then e, then f. The legend gives each thermal population.
The lower panel shows (point − curve) / error for each state. The grey band is ±1.96, the 95% range of a standard normal residual. Residuals that keep one sign over a range of delays show where the curves miss the points.
The title gives T1, the secondary time and the three lack-of-fit checks as J/df:

- `within-point`: the free three-state model at each delay, with p.
- `sharing`: shared readout nuisances against fully separate delays; it has no p.
- `curve`: the rate-equation curves against a free operation at each delay, with p.

The summary reports `t1` and `t1_err`, `secondary_time` and `secondary_time_err` in us, the three checks, and the joint `J` and `df`.
Errors are Wald errors from the joint fit. At low state contrast and few cycles per delay they can understate the uncertainty.
A fit failure in either step produces an analysis failure; the live plot and the saved sums remain.

This experiment proposes no writeback. Calibrate `$qubit.t1` with `twotone/t1`.

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

`chain_t1` takes `max_delay_us`, `points`, `reps`, `rounds`, `pi_ref` and `readout_ref`; points maps to delay Expts.
Set Relax delay with an override on `relax_delay`.
The recipe runs, analyzes and saves. It never asks for writeback.
The report gives `t1_us`, `t1_err_us`, `secondary_time_us`, `secondary_time_err_us` and the checks `within_point`, `sharing` and `curve` (each J, df, p).
Omitted inputs retain Form values; overrides apply last, then check_ready reports missing or invalid fields before Run.
