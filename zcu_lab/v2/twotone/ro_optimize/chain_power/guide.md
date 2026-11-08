# Readout gain by the readout-chain metric

**Readers:** operator, agent. Run the readout-chain gain sweep from the GUI or MCP and read its metric, d/sigma and λ per gain.

**Last updated:** 2026-11-09

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Behavior

`twotone/ro_optimize/chain_power` sweeps the readout gain. At each gain it reads the resonator 32 times in a row and repeats that block many times.
A shot is one block: 32 reads of the readout pulse, each read starting as soon as the previous read's pulse and ADC window have ended.
There is no reset, no qubit pulse and no relax delay. The sweep therefore needs no qubit pulse or qubit frequency, unlike `twotone/ro_optimize/power`, and can run before qubit calibration.
All Reps blocks of one gain run back to back before the next gain, so the blocks of one gain form one continuous read stream.
The experiment fixes the 32 reads per block and the zero relax delay; the Form does not offer them.

The host reduces each block to 113 moments of its IQ reads: means, second moments, correlations between reads 1 to 15 apart and third-order moments of reads 1 to 4 apart. Only running sums of these moments are kept.
From the sums, each gain gets a read-drive-read information metric. It measures how well a read, a drive that swaps the two most populated states, and a second read detect that swap, evaluated at equal populations of the two states. Larger is better.
The metric depends on how far apart the two states are in units of the read noise (d/sigma) and how often a read keeps the state (λ). It does not depend on the thermal population.

## Parameter sources

Calibrate the readout frequency first, for example with `onetone/freq`.
Readout uses the captured `$resonator.module.readout` slot, or a fresh `readout` template that reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly`.
Run replaces the readout gain with the sweep, so the Form hides the readout gain.
Gain starts at 10 points from 0.02 to 0.2 a.u. Reps starts at 1000 blocks per gain per round and Rounds at 20, which gives 20 000 blocks per gain.
Reps must be at least 2, because the metric needs two blocks per gain.
Each round reads every gain Reps × 32 times. One block lasts about 32 times the readout pulse length.
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

## Live plot

After every round the `measurement` plot shows the metric of each gain with its one-sigma error bar, from all finished rounds.
The metric takes about 10 ms per gain, so 10 gains take about 0.1 s per round. Acquisition waits while it is computed; then the GUI draws the plot.
The first computation in a GUI session can take about 6 s while Numba compiles the analysis code.
A gain whose metric cannot be computed is left out of the plot, and the plot title counts these settings.
Stop keeps the sums of finished rounds. A stop before the first round finishes keeps nothing, and the Run fails with that message.

## Analysis

Analysis recomputes the metric of each gain from the saved sums. It has no options.
The `metric` figure shows the metric with error bars and the best gain as a red dashed line. Its lower panel shows d/sigma of each gain.
The summary reports:

- `best`: the gain with the largest metric.
  - `setting`: that gain in a.u..
  - `metric` and `standard_error`: its metric and one-sigma error.
  - `separation`: its d/sigma.
  - `retention`: λ of its two most populated states, most populated first.
  - `third_population`: the population of the third fitted state, a leakage diagnostic.
- `failures`: each gain that could not be fitted, with its `setting` in a.u. and the `reason`, for example no identifiable readout signal or a singular matrix. These gains are left out of both panels. The figure title counts them.

If no gain has a metric, the analysis fails and lists all reasons.

A gain with no readout contrast can show a metric comparable to a real one, with a small error bar. Check the d/sigma panel to see which gains have signal before you choose a gain.
The error bar is the statistical error of the one-step estimate. It does not cover the estimator's bias. When the third state is not populated, the two λ values may not be separately identifiable.

This experiment proposes no writeback. Enter the chosen gain in the readout module yourself.

## Saved data

The saved record is the `ReadoutSums` of the run: the gain axis in a.u., the block count of each gain and the 113 moment sums of each gain.
Loading the file and analyzing it gives the same summary as the analysis of the run.
The native tag is `twotone/ro_optimize/chain_power` with cfg schema `3.0`.

## MCP recipe

`ro_optimize_chain_power` takes optional `gain_start` and `gain_stop` in a.u., `points` (gain Expts), `reps` and `rounds`. Omitted inputs keep Form values.
The recipe runs, analyzes and saves. It never asks for writeback.
The report gives `gain`, `metric`, `metric_err`, `separation`, `retention`, `third_population` and `failures`.
Overrides apply last, then check_ready reports missing or invalid fields before Run.
