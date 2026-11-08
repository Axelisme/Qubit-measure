# K-cycle readout-chain amplitude Rabi

**Readers:** operator, agent. Run the readout-chain gain Rabi from the GUI or MCP and read its live curves, joint fit and checks.

**Last updated:** 2026-11-09

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Behavior

`chain/amp_rabi` sweeps the Rabi Pulse gain like `twotone/rabi/amp_rabi`, but every repetition is one cycle of 16 reads instead of one read.
A cycle waits Relax delay, then plays eight blocks of the gain-swept Rabi Pulse followed by two reads of the same Readout, with no reset inside the cycle.
Relax delay is the thermalisation wait K: set it to at least 5 T1, so every cycle starts in thermal equilibrium.
Each read waits for the previous ADC window to close, so the second read of a block starts after the first and the next Rabi Pulse starts after the second.
The gain loop runs outside the cycle. Reps sets the cycles at every gain in one round, and Rounds sets the number of rounds.

The host reduces each cycle to 65 numbers: the IQ of the 16 reads, their I², IQ, Q² summed over three read groups, and IQ products between read pairs summed over six groups set by the number of Rabi pulses and reads between them.
Only running sums of these numbers are kept, so memory and the saved file do not grow with the number of cycles.

## Parameter sources

The Form uses the `twotone/rabi/amp_rabi` sources: the x180 copy or a fresh `qub_probe` pulse, the `$resonator.module.readout` slot or a fresh `readout` template, and 51 gains from -0.3 to `min(1.0, 1.2 * x180 gain)`.
Readout must be a pulse readout; a mux readout module is rejected by the Form. The cycle has no Reset or Init Pulse.
Relax delay starts at 5 * `$qubit.t1` us, or 150 us when T1 is missing.
Reps starts at 100 and Rounds at 100. Reps must be at least 2, and points × (Reps − 1) must be at least 65 so the first round can estimate the noise of the 65 numbers.
One cycle lasts about Relax delay + 8 × (pulse length + 2 × readout length).
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

## Live plot

After every round the `measurement` plot shows two curves from all completed rounds, both in standard errors of the point means (sigma):

- `all 65 numbers`: the 65 numbers relative to their median over gains, whitened by the noise within a gain and projected onto the direction that changes most across the sweep.
- `first read`: the mean IQ of read 0 relative to its median, projected onto its principal axis; this is the familiar single-read Rabi curve.

The sign of the first curve follows the second, so both oscillate together. Stop keeps the sums of completed rounds. A stop before the first round finishes keeps nothing and the Run fails with that message.

## Analysis

Analysis needs no options. It first fits the first read's mean IQ with the `twotone/rabi/amp_rabi` fit and publishes that experiment's `fit` figure; its π gain x_π gives the starting frequency f = 1 / (2 x_π).
It then fits all 65 numbers at all gains jointly with a three-state model whose readout and population nuisances are shared across gains.
The drive response per state is: from g the pulse reaches e with probability sin²(π f x); from e it stays in e with cos²(π f x); the third state is not driven.
f is in cycles per gain unit, and the π gain is 1 / (2 f).

The `populations` plot shows, per gain, the first-read populations of a fit with a free drive at each gain, with Wald error bars, and the fitted model as lines.
State labels are arbitrary; the legend gives each state's fitted thermal population, and the largest one is normally g.
The title gives the π gain and the three lack-of-fit checks as J/df:

- `within-point`: the free three-state model at each gain, with p.
- `sharing`: shared readout nuisances against fully separate gains; it has no p.
- `curve`: the Rabi curve against a free drive at each gain, with p.

The summary reports `pi_amp` and `pi_amp_err`, `f` and `f_err`, the reliability `R` = 1 / (2π × gain span × f_err), the three checks, the joint `J` and `df`, and `first_read_pi_amp` from the first-read fit.
Errors are Wald errors from the joint fit. At low state contrast and few cycles per gain they can understate the uncertainty. R is a diagnostic without a threshold.
A fit failure in either step produces an analysis failure; the live plot and the saved sums remain.

This experiment proposes no writeback. Compare `pi_amp` with `first_read_pi_amp`, and calibrate x180 with `twotone/rabi/amp_rabi`.

## Saved data

The saved record contains the gain axis, the cycle count and the 65 sums at each gain, and one 65 × 65 product sum added over all gains.
The product sum feeds only the live plot. Loading the file and analyzing it again gives the same summary.

## MCP recipe

`chain_amp_rabi` takes `gain_range`, `points`, `reps`, `rounds`, `frequency_mhz`, `pulse_length_us`, `drive_ref` and `readout_ref`; points maps to gain Expts.
Set Relax delay with an override on `relax_delay`.
The recipe runs, analyzes and saves. It never asks for writeback.
The report gives `pi_gain`, `pi_gain_err`, `frequency_per_gain`, `frequency_per_gain_err`, `reliability`, the checks `within_point`, `sharing` and `curve` (each J, df, p), and `first_read_pi_gain`.
Omitted inputs retain Form values; overrides apply last, then check_ready reports missing or invalid fields before Run.
