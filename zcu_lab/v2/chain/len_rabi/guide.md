# K-cycle readout-chain length Rabi

**Readers:** operator, agent. Run the readout-chain length Rabi from the GUI or MCP and read its live curves, joint fit and checks.

**Last updated:** 2026-11-09

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Behavior

`chain/len_rabi` sweeps the Rabi Pulse length like `twotone/rabi/len_rabi`, but every repetition is one cycle of 16 reads instead of one read.
A cycle waits Relax delay, then plays eight blocks of the length-swept Rabi Pulse followed by two reads of the same Readout, with no reset inside the cycle.
Relax delay is the thermalisation wait K, including before the first cycle. Set it to at least 5 T1 for the model's thermal-equilibrium start.
After a readout pulse ends, its `post_delay` sets the wait before the second readout or the next Rabi Pulse.
The next pulse keeps its own pre-delay but may start before the previous ADC window closes. Set the timing so windows on the same ADC do not overlap.
The FPGA sweeps the length, so the Rabi Pulse must use a const or flat_top waveform; Run rejects other waveforms before acquisition.
The length loop runs outside the cycle. Reps sets the cycles at every length in one round, and Rounds sets the number of rounds.

The host reduces each cycle to 65 numbers: the IQ of the 16 reads, their I², IQ, Q² summed over three read groups, and IQ products between read pairs summed over six groups set by the number of Rabi pulses and reads between them.
For a fixed sweep, the saved sums and the accumulator across rounds have fixed size, independent of the number of cycles.
Temporary arrays within one acquisition round grow with points × Reps. Increasing Reps therefore uses more temporary memory; increasing Rounds does not enlarge those per-round arrays.

## Parameter sources

The Rabi Pulse starts from a fresh `qub_probe` const pulse with gain 1.0, not from x180, because x180 usually has a shaped waveform that the FPGA length sweep cannot stretch.
Its gain is an uncalibrated starting value; confirm it before Run.
The length sweep accepts const and flat_top drive waveforms.
Readout uses the `$resonator.module.readout` slot or a fresh `readout` template, and must be a pulse readout; a mux readout module is rejected by the Form.
The length runs from 0.03 µs to 4 × the x180 length (1.0 µs without x180) over 101 points. The cycle has no Reset or Init Pulse.
Relax delay starts at 5 * `$qubit.t1` µs, or 150 µs when T1 is missing.
Reps starts at 100 and Rounds at 100. Reps must be at least 66, so that even a single length gives the first round 65 cycles beyond its mean to estimate the noise of the 65 numbers.
One cycle lasts about Relax delay + 8 × (pulse length + 2 × readout length).
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

## Live plot

After every round the `measurement` plot shows two curves from all completed rounds, both in standard errors of the point means (sigma):

- `all 65 numbers`: the 65 numbers relative to their median over lengths, whitened by the noise within a length and projected onto the direction that changes most across the sweep.
- `first read`: the mean IQ of read 0 relative to its median, projected onto its principal axis; this is the familiar single-read Rabi curve.

The sign of the first curve follows the second, so both oscillate together. Stop keeps the sums of completed rounds. A stop before the first round finishes keeps nothing and the Run fails with that message.

## Analysis

Analysis needs no options. It first fits the first read's mean IQ with the `twotone/rabi/len_rabi` decaying-cosine fit and publishes that experiment's `fit` figure; its π length x_π gives the starting frequency f = 1 / (2 x_π).
It then fits all 65 numbers at all lengths jointly with a three-state model whose readout and population nuisances are shared across lengths.
The drive row for each starting state interpolates between two free population endpoints, D_i(x) = [1 − f_i(x)] × endpoint_i0 + f_i(x) × endpoint_i1.
The curve weights are sin²(π f x) for g, cos²(π f x) for e and sin²(π f x) for the third state.
The third-state row can vary with x. Equal endpoints give a flat row; distinct endpoints allow an oscillating response.
f is in MHz, or cycles per µs. The reported π length is 1 / (2 f) in a zero-phase model with no length offset or decay envelope.
Interpret this value as the π total length only for a const pulse.
Run still accepts flat_top, but its total length includes ramps whose area adds a length offset.
The zero-phase model does not account for that offset, so its reported half-period is not a calibrated flat_top π total length.

The `populations` plot shows, per length, the first-read populations of a fit with a free drive at each length, with Wald error bars, and the fitted model as lines.
State labels are arbitrary; the legend gives each state's fitted thermal population, and the largest one is normally g.
The title gives the π length and the three lack-of-fit checks as J/df:

- `within-point`: the free three-state model at each length, with p.
- `sharing`: shared readout nuisances against fully separate lengths; it has no p.
- `curve`: the Rabi curve against a free drive at each length, with p. A decaying Rabi oscillation raises it.

The summary reports `pi_len` and `pi_len_err` in µs, `f` and `f_err` in MHz, the reliability `R` = 1 / (2π × length span × f_err), the three checks, the joint `J` and `df`, and `first_read_pi_len` from the first-read fit.
Errors are Wald errors from the joint fit. At low state contrast and few cycles per length they can understate the uncertainty. R is a diagnostic without a threshold.
A fit failure in either step produces an analysis failure; the live plot and the saved sums remain.

This experiment proposes no writeback. Compare `pi_len` with `first_read_pi_len`, and calibrate x180 with `twotone/rabi/len_rabi`.

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

The saved record contains the length axis, the cycle count and the 65 sums at each length, and one 65 × 65 product sum added over all lengths.
Lengths are us in memory and seconds in the file. The product sum feeds only the live plot. Loading the file and analyzing it again gives the same summary.

## MCP recipe

`chain_len_rabi` takes `max_length_us`, `points`, `reps`, `rounds`, `frequency_mhz`, `gain`, `drive_ref` and `readout_ref`; points maps to length Expts.
Set Relax delay with an override on `relax_delay`.
The recipe runs, analyzes and saves. It never asks for writeback.
The report gives `pi_length_us`, `pi_length_err_us`, `rabi_frequency_mhz`, `rabi_frequency_err_mhz`, `reliability`, the checks `within_point`, `sharing` and `curve` (each J, df, p), and `first_read_pi_length_us`.
Omitted inputs retain Form values; overrides apply last, then check_ready reports missing or invalid fields before Run.
