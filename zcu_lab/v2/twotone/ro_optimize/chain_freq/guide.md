# Readout frequency by the readout-chain metric

**Readers:** operator, agent. Run the readout-chain frequency sweep from the GUI or MCP and read its information, QND, d/sigma and λ per frequency.

**Last updated:** 2026-11-09

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Behavior

`twotone/ro_optimize/chain_freq` sweeps the readout frequency. The generator and the ADC both use the swept frequency. At each frequency it reads the resonator 32 times in a row and repeats that block many times.
A shot is one block: 32 reads of the readout pulse, each read starting as soon as the previous read's pulse and ADC window have ended.
There is no reset, no qubit pulse and no relax delay. The sweep therefore needs no qubit pulse or qubit frequency, unlike `twotone/ro_optimize/freq`, and can run before qubit calibration.
All Reps blocks of one frequency run back to back before the next frequency, so the blocks of one frequency form one continuous read stream.
The experiment fixes the 32 reads per block and the zero relax delay; the Form does not offer them.

The host reduces each block to 113 moments of its IQ reads: means, second moments, correlations between reads 1 to 15 apart and third-order moments of reads 1 to 4 apart. Only running sums of these moments are kept.
From the sums, each frequency gets an information value per read about a weak drive in an infinite symmetric ...DRDRDR... chain. D is the drive and R a read. The information combines products of reads at every lag. It uses the two most populated fitted states at equal populations. Larger is better. The report calls this number `metric`.
Information depends on the pair's QND and SNR. Here SNR is d/sigma, the distance between the states divided by the read noise. QND is R̃_aa + R̃_bb − 1 of the states' transition block, with each row renormalised within the pair. It lies between 0 and 1. At the limit of 1, reads never change the state within the pair. Information evaluates the pair at equal populations rather than its measured thermal population.
Information weighs QND more than SNR. At d/sigma 3 and QND 0.90, raising QND to 0.95 raises information from 9.6 to 24.5 (+154%), and raising d/sigma to 4 raises it to 11.0 (+14%).
SNR keeps a non-zero weight. Information is 0 at d/sigma 0 and rises with d/sigma at every QND, up to the limit 4·QND²/(1 − QND²).

## Parameter sources

Calibrate the resonator first, for example with `onetone/freq`, so `$resonator.freq` and `$resonator.kappa` exist.
Readout uses the captured `$resonator.module.readout` slot, or a fresh `readout` template that reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly`. The template gain 0.1 a.u. is an uncalibrated starting value; confirm it before Run.
Run replaces the readout frequency with the sweep, so the Form hides the readout frequency.
Readout freq starts at 11 points over `$resonator.freq` ± `$resonator.kappa` in MHz. Reps starts at 1000 blocks per frequency per round and Rounds at 20, which gives 20 000 blocks per frequency.
Reps must be at least 2, because the metric needs two blocks per frequency.
Each round reads every frequency Reps × 32 times. One block lasts about 32 times the readout pulse length.
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

## Live plot

After every round the `measurement` plot shows the information per read of each frequency with its one-sigma error bar, from all finished rounds.
The y axis is `Information per read (all lags)`.
Acquisition waits for the fits; then the GUI draws the plot. The first computation in a GUI session also waits for Numba to compile the analysis code.
Frequencies with no signal and frequencies whose fit failed are left off the curve. The plot title counts them apart, for example `Not plotted: 2 no signal, 1 fit failed`.
Stop keeps the sums of finished rounds. A stop before the first round finishes keeps nothing, and the Run fails with that message.

## Analysis

Analysis recomputes the information of each frequency from the saved sums. It has no options.
The upper panel of the `metric` figure shows the information with error bars and the best frequency as a red dashed line. The legend names the rule: the best frequency has the largest information minus its one-sigma error, so it need not be the highest point of the curve.
The lower panel shows QND of each frequency with its one-sigma error bar. d/sigma is not plotted; it is in the summary and the report.
The title counts the frequencies left off both panels, no-signal and failed fits apart.
The summary reports:

- `best`: the frequency with the largest information minus its one-sigma error.
  - `setting`: that frequency in MHz.
  - `metric` and `standard_error`: its information per read and one-sigma error.
  - `qnd` and `qnd_error`: its QND and one-sigma error.
  - `separation`: its d/sigma.
  - `retention`: λ of its two most populated states, most populated first.
  - `third_population`: the population of the third fitted state, a leakage diagnostic.
- `no_signal`: the frequencies, in MHz, that the no-signal test stopped before the fit.
- `failures`: each frequency whose fit failed, with its `setting` in MHz and the `reason`.

If no frequency is fitted, the analysis fails and states how many frequencies had no signal and how many fits failed, with the failure reasons.

"No signal" and a failed fit are different results.
A frequency has no signal when the test run before the fit finds too little read-to-read correlation in the moments of lags 1 to 4: the test statistic stays below 2·ln(blocks), or the largest eigenvalue of the lag covariance is at most 0. Such a frequency is not fitted and not ranked. More blocks can lift a weak frequency above the test.
A failure means the estimator could not produce a fit, rather than classifying the setting as no signal. Causes include a singular matrix, a non-finite result, or the fit reaching its budget of 1200 evaluations. The `reason` says which.

If the best frequency is the first or last frequency of the scan, the optimum can lie outside the scanned range. Extend the range on that side and run again before you use the frequency.

The error bars are the statistical errors of the fit. They do not cover the fit's bias. When the third state is not populated, the two λ values may not be separately identifiable.

This experiment proposes no writeback. Enter the chosen frequency in the readout module yourself.

## Saved data

The saved record is the `ReadoutSums` of the run: the frequency axis (MHz in memory, Hz in the file), the block count of each frequency and the 113 moment sums of each frequency.
Loading the file and analyzing it gives the same summary as the analysis of the run.
The native tag is `twotone/ro_optimize/chain_freq` with cfg schema `3.0`.

## MCP recipe

`ro_optimize_chain_freq` takes optional `freq_start_mhz` and `freq_stop_mhz`, `points` (frequency Expts), `reps` and `rounds`. Omitted inputs keep Form values.
The recipe runs, analyzes and saves. It never asks for writeback.
The report gives `freq_mhz` (the best frequency by information minus one sigma), `metric`, `metric_err`, `qnd`, `qnd_err`, `separation`, `retention`, `third_population`, `no_signal` and `failures`.
Overrides apply last, then check_ready reports missing or invalid fields before Run.
