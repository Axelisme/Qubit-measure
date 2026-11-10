# Readout length by the R-pi alternating-contrast score

**Readers:** operator, agent. Run the R-pi ADC window sweep from the GUI or RPC and read its score, diagnostics and invalid reasons per window.

**Last updated:** 2026-11-10

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Behavior

`twotone/ro_optimize/rpi_length` sweeps the readout ADC window in us with a calibrated π pulse. At each window it plays one chain `...R π R π...` with no reset or relax delay, where R is the readout and π is the same fixed pulse at every window.
Use it next to `twotone/ro_optimize/length` and `twotone/ro_optimize/chain_length`. The first compares g and e prepared with a π pulse, the second uses no π pulse and waits for thermal relaxation between shots, and this one alternates reads with π pulses. Each is a separate experiment.

The ADC window and the readout pulse are different lengths. The ADC window is the swept value, rounded to the ADC clock. The readout pulse is that window plus 0.11 us, rounded to the generator clock. The host builds one program per window, so a short window gets a short pulse and not the longest pulse of the scan.
The π frequency, gain and length do not follow the sweep, and neither do the readout gain, frequency or `post_delay`.

One rep is 16 R-π pairs. All Reps of one window run before the next window starts, with relax delay 0. One round of one window is one chain of Reps × 16 reads, and lags up to 15 run across reps.
Each round starts a new chain at every window. The analysis never pairs reads across a round or window boundary.
The experiment fixes the 16 pairs per rep, the zero relax delay and the 0.11 us margin. The Form does not offer them.

The wait from R to π is the readout's `post_delay` plus the π pulse's own `pre_delay`, which is usually 0. With `post_delay` 0.05 us and a π `pre_delay` of 0.02 us, each π starts 0.07 us after the readout generator pulse ends, and the wait is the same at every window. The wait counts from the generator end and not from the ADC window end, so a π can play while the ADC window is still open. Inside a rep, the next readout starts after the π ends, later only by the π pulse's `post_delay` and the readout's `pre_delay`, which are usually 0.
The Form has no separate wait field. Change `post_delay` in the readout module to change this wait. Run keeps every π setting, including a nonzero `pre_delay`.

Rep boundaries have different timing. Each rep ends with the shot closure, which waits for every ADC window to close, and the next rep starts after it. When the last ADC window closes after the last π ends, the step across a rep boundary is longer than a step inside a rep. The closure is not a relax delay, and the Form has no field for it. In an offline build with a 0.05 us post-delay, 0.04 us π and 0.25 us trigger offset, the closure added 0.05 us once per 16 pairs at every window.
Reads are therefore not guaranteed to be evenly spaced. The analysis still treats the reps of one window and round as one chain, with lags across reps.

Run does not reset the qubit, discard reads, add a warmup or calibrate the π pulse.

## Parameter sources

Calibrate the readout frequency and the π pulse first. The Form cannot check the π calibration, and a wrong π changes the score and every diagnostic.
Readout uses the captured `$resonator.module.readout` slot, or a fresh `readout` template that reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly`. The template starts at gain 0.1, which is an uncalibrated value to confirm.
Pi pulse uses the captured `$qubit.module.x180` slot, or a fresh `pi_pulse` template.
Run sets the ADC window from the sweep and the pulse length from the window, so the Form hides both lengths in the readout module.
ADC window is in us. It starts at 11 points from 0.2 to 2.2 us.
Reps starts at 1000 and Rounds at 20, which gives 16 000 reads per chain and 320 000 reads per window.
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

## Live plot

After every round the `measurement` plot shows the score of each valid window with a finished round, from all finished rounds. The x axis is `ADC window (us)` and the y axis is `Alternating-contrast score`.
Invalid windows are gaps in the curve. The title counts them, for example `Not plotted: 3 invalid`. Windows that have not finished a round are not on the curve and are not counted as invalid.

Stop keeps the sums of finished rounds. Windows with at least one finished round stay in the result, and windows that were not measured are dropped from it. A stop before any window finishes a round keeps nothing, and the Run fails with that message.

## Analysis

Analysis recomputes every window from its own saved sums. It has no options.
The `score` figure has six panels, all with `ADC window (us)` on the x axis:

- The score of each valid window. A red dashed line and the legend mark the best window in us.
- `Invalid reason per window`. Each row is a reason that occurred, with a mark at each window reporting it.
- `Q_eff`, `q`, amplitude `a` and noise variance `σ²`, for valid windows only.

The figure title counts the invalid windows. Invalid windows appear only in the reason panel. No panel draws `σ²` as an error bar on the score.

The summary reports:

- `best_length`: the valid ADC window with the largest score, in us.
- `best`: every estimate at that window, listed below.
- `length_count`: the number of windows in the saved result.
- `not_plotted`: the number of invalid windows.
- `invalid_reasons`: how many windows reported each reason.

The summary has a fixed size for any scan length, so it does not list each invalid window. The reason panel shows which window reported which reason, and the saved sums reproduce them.

If no window is valid, `best_length` and `best` are empty and the title says there is no valid window to select. The analysis does not fall back to a default window and does not fail.
If the best window is the first or last window of the scan, the optimum can lie outside the scanned range. Extend the range on that side and run again before you use the window.
This experiment proposes no writeback. Enter the chosen ADC window, and a pulse length of that window plus 0.11 us, in the readout module yourself.

### Score

The score is the alternating-contrast score, 交替對比分數:

`score = (T − 3.719 √V₀) / r₀`

`T` is the alternating contrast of the fixed lags 1 to 15. `V₀` is its variance under a white-noise model. `r₀` is the same-read variance divided by 2.
The best window is the valid window with the largest score. The analysis applies one score and no second selection threshold. `Q_eff`, `a` and `σ²` only decide validity and serve as diagnostics. They do not enter the score.

The score is a heuristic utility. It is not Fisher information, QND fidelity or MIST. In this guide MIST means leakage out of g and e only.

At high SNR the score prefers a stronger readout than the point with the most information, so the chosen window can differ from the information optimum. The table compares the finite-window information of the chosen point with the best finite-window information in the same scan. These numbers come from one two-state synthetic scan. They are not a general guarantee and no hardware measurement has checked them. Here `q = a / σ²`, which is not d/σ.

| Largest q in the scan | Chosen point / best point |
| --- | --- |
| 0.05 to 0.1 | 97 to 100% |
| 0.5 | 87 to 92% |
| 2 | 55% |

### Gate and invalid windows

A window passes the gate when `T − 3.719 √V₀ > 0.0224 r₀`, which is the same as `score > 0.0224`. The gate uses only that window's data and never another window or the scan range.
3.719 corresponds to a one-sided nominal α of 10⁻⁴ per window under white noise. It does not change with shots or the number of windows. 0.0224 is a fixed minimum effect and not a Form field.

The nominal α holds for white noise only. Correlated electronics noise can change the real false-pass rate. Passing shows an observable alternating contrast above the minimum effect. It does not prove the signal comes from the qubit and it does not exclude leakage. Systematic errors larger than the minimum effect are not covered.

A window is invalid when any of these reasons applies. Invalid windows keep their reasons, have no score and are never ranked.

| Reason | Meaning |
| --- | --- |
| `gate_failed` | The contrast did not pass the gate. |
| `q_eff_out_of_range` | `Q_eff` is outside 0 to 1. |
| `nonpositive_amplitude` | `a` is not positive. |
| `nonpositive_noise_variance` | `σ²` is not positive. |
| `zero_denominator` | A lag has no pairs or an estimate divides by zero. |
| `nonpositive_same_read_variance` | The same-read variance is not positive. |

This experiment has no separate no-signal test. A window with no alternating signal reports `gate_failed`. More reads shrink `V₀`, but a fixed offset below the minimum effect still does not pass.

### Estimates

Each estimate comes from a closed-form calculation on lags 1 to 15, with no fit, iteration or automatic cutoff.

- `q_eff` is the effective retention per R-π step. For a symmetric two-state readout with a π flip error ε, `Q_eff = Q (1 − 2ε)`. It includes the π error, so it is not the pure readout QND, and the analysis does not correct for ε.
- `amplitude` is `a`, the alternating-mode amplitude in ADC units².
- `noise_variance` is `σ²`, the noise variance of one quadrature in ADC units². It is not an error bar and the analysis reports no standard error.
- `q` is `a / σ²`. For a two-state readout it equals (d/σ)² / 4.
- `contrast`, `null_variance` and `same_read_variance` are `T`, `V₀` and `r₀`.
- `gate_passed` is the gate result, and `invalid_reasons` lists the reasons for that window.

Readout-chain analyses that start from a thermal state can combine a readout score with an M penalty. Every read of this stationary chain has the same statistics, so M has expectation 0. This experiment does not compute or save M and does not use that combined score.

### Limits

The estimates assume a stationary two-state chain. Several things can bias them:

- Starting population. A chain starts from whatever state the previous window or round left. No reset or warmup removes the start of the chain, and its reads count like all others.
- Third state. Population outside the two alternating states can bias every estimate. The analysis removes a slow mode as an approximation, which is not a leakage correction. This experiment does not identify general leakage and does not correct readout QND.
- Large IQ offsets and π errors can bias the estimates.
- Longer steps at rep boundaries. The estimates assume the same retention at every step. A rep boundary that waits for the ADC window is a longer step, so its retention can differ. The analysis does not correct or exclude this, and no one has measured the size of the resulting bias.
- A physical break in the chain that leaves no trace in the IQ data cannot be detected. Each finished round and window counts as one chain, and no hardware run has verified that it is unbroken.

The offline probe of this experiment checks the program and the data contract with synthetic data. It does not calibrate hardware and it does not validate the physical model.

## Saved data

The saved record has the ADC windows that finished at least one round, in us in memory and seconds on disk, and one 35-column `rpi_sums` record per window. A record holds the number of chains, the read count, the 18 moment sums and the 15 pair counts. Every chain of a window has the same length, so one count and one length describe the chains. Raw IQ is not saved.
Loading the file and analyzing it gives the same summary as the analysis of the run.
The native tag is `twotone/ro_optimize/rpi_length` with cfg schema `3.0`.

## MCP

This experiment has no MCP recipe. Use the GUI or the RPC tools.
