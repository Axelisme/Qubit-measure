# K-cycle readout-chain T1

**Readers:** operator, agent. Run the readout-chain T1 from the GUI or MCP and read its live curves, joint fit and checks.

**Last updated:** 2026-11-09

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Behavior

`chain/t1` sweeps the wait after a π pulse like `twotone/t1`, but every repetition is one cycle of 16 reads instead of one read.
A cycle waits Relax delay, then plays eight blocks of the Pi Pulse, the swept Delay and two reads of the same Readout, with no reset inside the cycle.
Relax delay is the thermalisation wait K: set it to at least 5 T1, so every cycle starts in thermal equilibrium.
Each read waits for the previous ADC window to close, so the second read of a block starts after the first and the next Pi Pulse starts after the second.
The delay loop runs outside the cycle. Reps sets the cycles at every delay in one round, and Rounds sets the number of rounds.

The host reduces each cycle to 65 numbers: the IQ of the 16 reads, their I², IQ, Q² summed over three read groups, and IQ products between read pairs summed over six groups set by the number of π pulses and reads between them.
Only running sums of these numbers are kept, so memory and the saved file do not grow with the number of cycles.

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

Analysis needs no options. It first fits the first read's mean IQ with the `twotone/t1` single-decay fit and publishes that experiment's `fit` figure; its T1 starts both fitted times.
It then fits all 65 numbers at all delays jointly with a three-state model whose readout and population nuisances are shared across delays.
The response per state is: from g the π pulse reaches e, which decays back with exp(−x / t1); from e the π pulse lands in g, which does not decay, so that row is nearly flat; the third state decays with its own exp(−x / tf).
Both times are in us. The third state's decay only follows exp(−x / tf) when it decays straight to one state; a cascade through e makes tf an effective value.

The `populations` plot shows, per delay, the first-read populations of a fit with a free operation at each delay, with Wald error bars, and the fitted model as lines.
State labels are arbitrary; the legend gives each state's fitted thermal population, and the largest one is normally g.
The title gives T1, tf and the three lack-of-fit checks as J/df:

- `within-point`: the free three-state model at each delay, with p.
- `sharing`: shared readout nuisances against fully separate delays; it has no p.
- `curve`: the decay curves against a free operation at each delay, with p.

The summary reports `t1` and `t1_err`, `tf` and `tf_err` in us, the three checks, the joint `J` and `df`, and `first_read_t1` from the first-read fit.
Errors are Wald errors from the joint fit. At low state contrast and few cycles per delay they can understate the uncertainty. A small third-state population leaves tf poorly determined.
A fit failure in either step produces an analysis failure; the live plot and the saved sums remain.

This experiment proposes no writeback. Compare `t1` with `first_read_t1`, and calibrate `$qubit.t1` with `twotone/t1`.

## Saved data

The saved record contains the delay axis, the cycle count and the 65 sums at each delay, and one 65 × 65 product sum added over all delays.
Delays are us in memory and seconds in the file. The product sum feeds only the live plot. Loading the file and analyzing it again gives the same summary.

## MCP recipe

`chain_t1` takes `max_delay_us`, `points`, `reps`, `rounds`, `pi_ref` and `readout_ref`; points maps to delay Expts.
Set Relax delay with an override on `relax_delay`.
The recipe runs, analyzes and saves. It never asks for writeback.
The report gives `t1_us`, `t1_err_us`, `tf_us`, `tf_err_us`, the checks `within_point`, `sharing` and `curve` (each J, df, p), and `first_read_t1_us`.
Omitted inputs retain Form values; overrides apply last, then check_ready reports missing or invalid fields before Run.
