# Readout length by the readout-chain metric

**Readers:** operator, agent. Run the readout-chain length sweep from the GUI or MCP and read its metric, d/sigma and λ per ADC window.

**Last updated:** 2026-11-09

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Behavior

`twotone/ro_optimize/chain_length` sweeps the readout length: the ADC integration window and, with it, the readout pulse. At each window it reads the resonator 32 times in a row and repeats that block many times.
A shot is one block: 32 reads of the readout pulse, each read starting as soon as the previous read's pulse and ADC window have ended.
At each window the readout pulse is that window plus 0.11 us, the same margin as `twotone/ro_optimize/length`. Each setting therefore reads for its own duration, so λ can change with the read length.
There is no reset, no qubit pulse and no relax delay. The sweep therefore needs no qubit pulse or qubit frequency, unlike `twotone/ro_optimize/length`, and can run before qubit calibration.
As in `twotone/ro_optimize/length`, the host builds one program per window and runs all rounds of a window before the next window. The Reps blocks of a window run back to back, so they form one continuous read stream.
The experiment fixes the 32 reads per block and the zero relax delay; the Form does not offer them.

The host reduces each block to 113 moments of its IQ reads: means, second moments, correlations between reads 1 to 15 apart and third-order moments of reads 1 to 4 apart. Only running sums of these moments are kept.
From the sums, each window gets a read-drive-read information metric. It measures how well a read, a drive that swaps the two most populated states, and a second read detect that swap, evaluated at equal populations of the two states. Larger is better.
The metric depends on how far apart the two states are in units of the read noise (d/sigma) and how often a read keeps the state (λ). It does not depend on the thermal population.
The metric is per read, not per unit time. A longer window also makes each read and each block longer.

## Parameter sources

Calibrate the readout frequency first, for example with `onetone/freq`.
Readout uses the captured `$resonator.module.readout` slot, or a fresh `readout` template that reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly`. The template gain 0.1 a.u. is an uncalibrated starting value; confirm it before Run.
Run sets the ADC window from the sweep and the pulse length to that window plus 0.11 us, so the Form hides both.
Readout length starts at 11 points from 0.2 to 2.2 us. Reps starts at 1000 blocks per window per round and Rounds at 20, which gives 20 000 blocks per window.
Reps must be at least 2, because the metric needs two blocks per window.
Each round reads one window Reps × 32 times. One block lasts about 32 times that window's pulse length.
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

## Live plot

After every round the `measurement` plot shows the metric of each finished window with its one-sigma error bar, from all finished rounds. Windows not measured yet are not shown.
The metric takes about 10 ms per window, so 11 windows take about 0.1 s per round. Acquisition waits while it is computed; then the GUI draws the plot.
The first computation in a GUI session can take about 6 s while Numba compiles the analysis code.
A window whose metric cannot be computed is left out of the plot, and the plot title counts these settings.
Stop keeps every window that finished at least one round. A stop before the first window finishes a round keeps nothing, and the Run fails with that message.

## Analysis

Analysis recomputes the metric of each window from the saved sums. It has no options.
The `metric` figure shows the metric with error bars and the best window as a red dashed line. Its lower panel shows d/sigma of each window.
The summary reports:

- `best`: the window with the largest metric.
  - `setting`: that ADC window in us; its pulse is 0.11 us longer.
  - `metric` and `standard_error`: its metric and one-sigma error.
  - `separation`: its d/sigma.
  - `retention`: λ of its two most populated states, most populated first.
  - `third_population`: the population of the third fitted state, a leakage diagnostic.
- `failures`: each window that could not be fitted, with its `setting` in us and the `reason`, for example no identifiable readout signal or a singular matrix. These windows are left out of both panels. The figure title counts them.

If no window has a metric, the analysis fails and lists all reasons.

A window with no readout contrast can show a metric comparable to a real one, with a small error bar. Check the d/sigma panel to see which windows have signal before you choose a window.
The error bar is the statistical error of the one-step estimate. It does not cover the estimator's bias. When the third state is not populated, the two λ values may not be separately identifiable.

This experiment proposes no writeback. Enter the chosen window, and a pulse length 0.11 us longer, in the readout module yourself.

## Saved data

The saved record is the `ReadoutSums` of the run: the window axis (us in memory, seconds in the file), the block count of each window and the 113 moment sums of each window. A stopped run saves only the windows that finished a round.
Loading the file and analyzing it gives the same summary as the analysis of the run.
The native tag is `twotone/ro_optimize/chain_length` with cfg schema `3.0`.

## MCP recipe

`ro_optimize_chain_length` takes optional `length_start_us` and `length_stop_us`, `points` (window Expts), `reps` and `rounds`. Omitted inputs keep Form values.
The recipe runs, analyzes and saves. It never asks for writeback.
The report gives `length_us`, `metric`, `metric_err`, `separation`, `retention`, `third_population` and `failures`.
Overrides apply last, then check_ready reports missing or invalid fields before Run.
