# Readout frequency by the readout-chain metric

**Readers:** operator, agent. Run the readout-chain frequency sweep from the GUI or MCP and read its metric, d/sigma and λ per frequency.

**Last updated:** 2026-11-09

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Behavior

`twotone/ro_optimize/chain_freq` sweeps the readout frequency. The generator and the ADC both use the swept frequency. At each frequency it reads the resonator 32 times in a row and repeats that block many times.
A shot is one block: 32 reads of the readout pulse, each read starting as soon as the previous read's pulse and ADC window have ended.
There is no reset, no qubit pulse and no relax delay. The sweep therefore needs no qubit pulse or qubit frequency, unlike `twotone/ro_optimize/freq`, and can run before qubit calibration.
All Reps blocks of one frequency run back to back before the next frequency, so the blocks of one frequency form one continuous read stream.
The experiment fixes the 32 reads per block and the zero relax delay; the Form does not offer them.

The host reduces each block to 113 moments of its IQ reads: means, second moments, correlations between reads 1 to 15 apart and third-order moments of reads 1 to 4 apart. Only running sums of these moments are kept.
From the sums, each frequency gets a read-drive-read information metric. It measures how well a read, a drive that swaps the two most populated states, and a second read detect that swap, evaluated at equal populations of the two states. Larger is better.
The metric depends on how far apart the two states are in units of the read noise (d/sigma) and how often a read keeps the state (λ). It does not depend on the thermal population.

## Parameter sources

Calibrate the resonator first, for example with `onetone/freq`, so `$resonator.freq` and `$resonator.kappa` exist.
Readout uses the captured `$resonator.module.readout` slot, or a fresh `readout` template that reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly`. The template gain 0.1 a.u. is an uncalibrated starting value; confirm it before Run.
Run replaces the readout frequency with the sweep, so the Form hides the readout frequency.
Readout freq starts at 11 points over `$resonator.freq` ± `$resonator.kappa` in MHz. Reps starts at 1000 blocks per frequency per round and Rounds at 20, which gives 20 000 blocks per frequency.
Reps must be at least 2, because the metric needs two blocks per frequency.
Each round reads every frequency Reps × 32 times. One block lasts about 32 times the readout pulse length.
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

## Live plot

After every round the `measurement` plot shows the metric of each frequency with its one-sigma error bar, from all finished rounds.
The metric takes about 10 ms per frequency, so 11 frequencies take about 0.1 s per round. Acquisition waits while it is computed; then the GUI draws the plot.
The first computation in a GUI session can take about 6 s while Numba compiles the analysis code.
A frequency whose metric cannot be computed is left out of the plot, and the plot title counts these settings.
Stop keeps the sums of finished rounds. A stop before the first round finishes keeps nothing, and the Run fails with that message.

## Analysis

Analysis recomputes the metric of each frequency from the saved sums. It has no options.
The `metric` figure shows the metric with error bars and the best frequency as a red dashed line. Its lower panel shows d/sigma of each frequency.
The summary reports:

- `best`: the frequency with the largest metric.
  - `setting`: that frequency in MHz.
  - `metric` and `standard_error`: its metric and one-sigma error.
  - `separation`: its d/sigma.
  - `retention`: λ of its two most populated states, most populated first.
  - `third_population`: the population of the third fitted state, a leakage diagnostic.
- `failures`: each frequency that could not be fitted, with its `setting` in MHz and the `reason`, for example no identifiable readout signal or a singular matrix. These frequencies are left out of both panels. The figure title counts them.

If no frequency has a metric, the analysis fails and lists all reasons.

A frequency with no readout contrast can show a metric comparable to a real one, with a small error bar. Check the d/sigma panel to see which frequencies have signal before you choose a frequency.
The error bar is the statistical error of the one-step estimate. It does not cover the estimator's bias. When the third state is not populated, the two λ values may not be separately identifiable.

This experiment proposes no writeback. Enter the chosen frequency in the readout module yourself.

## Saved data

The saved record is the `ReadoutSums` of the run: the frequency axis (MHz in memory, Hz in the file), the block count of each frequency and the 113 moment sums of each frequency.
Loading the file and analyzing it gives the same summary as the analysis of the run.
The native tag is `twotone/ro_optimize/chain_freq` with cfg schema `3.0`.

## MCP recipe

`ro_optimize_chain_freq` takes optional `freq_start_mhz` and `freq_stop_mhz`, `points` (frequency Expts), `reps` and `rounds`. Omitted inputs keep Form values.
The recipe runs, analyzes and saves. It never asks for writeback.
The report gives `freq_mhz`, `metric`, `metric_err`, `separation`, `retention`, `third_population` and `failures`.
Overrides apply last, then check_ready reports missing or invalid fields before Run.
