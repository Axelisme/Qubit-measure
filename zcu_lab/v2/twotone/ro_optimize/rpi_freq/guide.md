# Readout frequency by the R-pi alternating-contrast score

**Readers:** operator, agent. Run the R-pi readout frequency sweep from the GUI or RPC and read its score, diagnostics and invalid reasons per frequency.

**Last updated:** 2026-11-10

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Behavior

`twotone/ro_optimize/rpi_freq` sweeps the readout frequency in MHz with a calibrated π pulse. At each frequency it plays cycles of `...R π R π...` with no reset or relax delay, where R is the readout and π is the same fixed pulse at every frequency.
Use it next to `twotone/ro_optimize/freq` and `twotone/ro_optimize/chain_freq`. The first compares g and e prepared with a π pulse, the second uses no π pulse and waits for thermal relaxation between shots, and this one alternates reads with π pulses. Each is a separate experiment.

The sweep sets the generator and ADC frequency of the readout together. The π frequency, gain and length do not follow the sweep, and neither do the readout gain, length or `post_delay`.
With `mixer_freq` set on the readout, the sweep and the saved frequency axis are absolute RF in MHz, on the common generator and ADC grid. The DDS plays RF minus the rounded mixer frequency. Without `mixer_freq`, the values are the DDS frequency. An interpolating generator rejects a frequency outside its DDS range and Run fails before acquisition. With a mixer, that range applies to RF minus the rounded mixer frequency.

One rep is one cycle of 64 R-π pairs. All Reps of one frequency run before the next frequency starts, with relax delay 0.
Lags 1 to 15 pair reads only within the same cycle. The statistics accumulate across reps and rounds, without pairs crossing a rep, round or frequency boundary.
The experiment fixes the 64 pairs per rep and the zero relax delay. The Form does not offer them.

The wait from R to π is the readout's `post_delay` plus the π pulse's own `pre_delay`, which is usually 0. With `post_delay` 0.05 µs and a π `pre_delay` of 0.02 µs, each π starts 0.07 µs after the readout generator pulse ends, and the wait is the same at every frequency. The wait counts from the generator end and not from the ADC window end, so a π can play while the ADC window is still open. Inside a rep, the next readout starts after the π ends, later only by the π pulse's `post_delay` and the readout's `pre_delay`, which are usually 0.
The Form has no separate wait field. Change `post_delay` in the readout module to change this wait. Run keeps every π setting, including a nonzero `pre_delay`.

Each rep ends with the shot closure, which waits for every ADC window to close before the next rep starts. If the last ADC window closes after the last π ends, this adds a wait at the rep boundary. The closure is not a relax delay, and the Form has no field for it.
This rep-boundary wait does not appear in a lag pair's time interval, but it can still affect the estimate through the next cycle's starting population.
If one ADC window would overlap the next, Run fails when the program builds and names the ADC overlap. Shorten the ADC window or lengthen the readout or its post-delay.

Run does not reset the qubit, discard reads, add a warmup or calibrate the π pulse.

## Parameter sources

Calibrate the π pulse first. The Form cannot check the π calibration, and a wrong π changes the score and every diagnostic.
Readout uses the captured `$resonator.module.readout` slot, or a fresh `readout` template that reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly`. The template starts at gain 0.1, which is an uncalibrated value to confirm.
Pi pulse uses the captured `$qubit.module.x180` slot, or a fresh `pi_pulse` template.
Run replaces the readout frequency with the sweep, so the Form hides the readout frequency.
Frequency is in MHz. It starts at 11 points around `$resonator.freq`, from `$resonator.freq` minus `$resonator.kappa` to `$resonator.freq` plus `$resonator.kappa`.
Reps starts at 250 and Rounds at 20. Each frequency has 5 000 cycles of 64 reads, or 320 000 reads in total.
Missing or invalid sources leave the affected fields empty and Invalid; supply them before Run.

## Live plot

After every round the `measurement` plot shows the score of each valid frequency, from all finished rounds. The x axis is `Readout frequency (MHz)` and the y axis is `Alternating-contrast score`.
Invalid frequencies are gaps in the curve. The title counts them, for example `Not plotted: 3 invalid`.
Stop keeps the sums of finished rounds. A stop before the first round finishes keeps nothing, and the Run fails with that message.

## Analysis

Analysis recomputes every frequency from its own saved sums. It has no options.
The `score` figure has six panels, all with `Readout frequency (MHz)` on the x axis:

- The score of each valid frequency. A red dashed line and the legend mark the best frequency in MHz.
- `Invalid reason per frequency`. Each row is a reason that occurred, with a mark at each frequency reporting it.
- `Q_eff`, `q`, amplitude `a` and noise variance `σ²`, for valid frequencies only.

The figure title counts the invalid frequencies. Invalid frequencies appear only in the reason panel.

The summary reports:

- `best_freq`: the valid frequency with the largest score, in MHz.
- `best`: every estimate at that frequency, listed below.
- `freq_count`: the number of frequencies in the sweep.
- `not_plotted`: the number of invalid frequencies.
- `invalid_reasons`: how many frequencies reported each reason.

The summary has a fixed size for any scan length, so it does not list each invalid frequency. The reason panel shows which frequency reported which reason, and the saved sums reproduce them.

If no frequency is valid, `best_freq` and `best` are empty and the title says there is no valid frequency to select. The analysis does not fall back to a default frequency and does not fail.
If the best frequency is the first or last frequency of the scan, the optimum can lie outside the scanned range. Extend the range on that side and run again before you use the frequency.
This experiment proposes no writeback. Enter the chosen frequency in the readout module yourself.

### Score

The score is the alternating-contrast score, 交替對比分數:

`score = (T − 3.719 √V₀) / r₀`

`T` is the alternating contrast of the fixed lags 1 to 15. `V₀` is its variance under a white-noise model. `r₀` is the same-read variance divided by 2.
The best frequency is the valid frequency with the largest score. The analysis applies one score and no second selection threshold. `Q_eff`, `a` and `σ²` only decide validity and serve as diagnostics. They do not enter the score.

The score is a heuristic utility. It is not Fisher information, QND fidelity or MIST. In this guide MIST means leakage out of g and e only.

At high SNR the score prefers a stronger readout than the point with the most information, so the chosen frequency can differ from the information optimum. The table compares the finite-window information of the chosen point with the best finite-window information in the same scan. These numbers come from one two-state synthetic scan. They are not a general guarantee and no hardware measurement has checked them. Here `q = a / σ²`, which is not d/σ.

| Largest q in the scan | Chosen point / best point |
| --- | --- |
| 0.05 to 0.1 | 97 to 100% |
| 0.5 | 87 to 92% |
| 2 | 55% |

### Gate and invalid frequencies

A frequency passes the gate when `T − 3.719 √V₀ > 0.0224 r₀`, which is the same as `score > 0.0224`. The gate uses only that frequency's data and never another frequency or the scan range.
3.719 corresponds to a one-sided nominal α of 10⁻⁴ per frequency under white noise. It does not change with shots or the number of frequencies. 0.0224 is a fixed minimum effect and not a Form field.

The nominal α holds for white noise only. Correlated electronics noise can change the real false-pass rate. Passing shows an observable alternating contrast above the minimum effect. It does not prove the signal comes from the qubit and it does not exclude leakage. Systematic errors larger than the minimum effect are not covered.

A frequency is invalid when any of these reasons applies. Invalid frequencies keep their reasons, have no score and are never ranked.

| Reason | Meaning |
| --- | --- |
| `gate_failed` | The contrast did not pass the gate. |
| `q_eff_out_of_range` | `Q_eff` is outside 0 to 1. |
| `nonpositive_amplitude` | `a` is not positive. |
| `nonpositive_noise_variance` | `σ²` is not positive. |
| `zero_denominator` | A lag has no pairs or an estimate divides by zero. |
| `nonpositive_same_read_variance` | The same-read variance is not positive. |

This experiment has no separate no-signal test. A frequency with no alternating signal reports `gate_failed`. More reads shrink `V₀`, but a fixed offset below the minimum effect still does not pass.

### Estimates

Each estimate comes from a closed-form calculation on lags 1 to 15, with no fit, iteration or automatic cutoff.

- `q_eff` is the effective retention per R-π step. For a symmetric two-state readout with a π flip error ε, `Q_eff = Q (1 − 2ε)`. It includes the π error, so it is not the pure readout QND, and the analysis does not correct for ε.
- `amplitude` is `a`, the alternating-mode amplitude in ADC units².
- `noise_variance` is `σ²`, the noise variance of one quadrature in ADC units². It is not an error bar and the analysis reports no standard error.
- `q` is `a / σ²`. For a two-state readout it equals (d/σ)² / 4.
- `contrast`, `null_variance` and `same_read_variance` are `T`, `V₀` and `r₀`.
- `gate_passed` is the gate result, and `invalid_reasons` lists the reasons for that frequency.

Readout-chain analyses that start from a thermal state can combine a readout score with an M penalty. Every read of this stationary chain has the same statistics, so M has expectation 0. This experiment does not compute or save M and does not use that combined score.

### Limits

The estimates assume a stationary two-state chain. Several things can bias them:

- Starting population. Each cycle starts from whatever state the preceding cycle or setting left. No reset or warmup removes the start of the cycle, and its reads count like all others.
- Third state. Population outside the two alternating states can bias every estimate. The analysis removes a slow mode as an approximation, which is not a leakage correction. This experiment does not identify general leakage and does not correct readout QND.
- Large IQ offsets and π errors can bias the estimates.
- Slow drift. All reps of one setting run together. Drift can become a difference between settings; multiple rounds average over these changes.
- A physical break inside a cycle that leaves no trace in the IQ data cannot be detected. No lag pair crosses a break between reps, but the break can change the next cycle's starting population. No hardware run has verified continuity inside a 64-read cycle.

The offline probe of this experiment checks the program and the data contract with synthetic data. It does not calibrate hardware and it does not validate the physical model.

## Saved data

The saved record has the frequencies, in MHz in memory and Hz on disk, and one 35-column `rpi_sums` record per frequency. A record holds the completed cycle count S, the read count N, the 18 moment sums and the 15 pair counts. Each cycle has 64 reads, so L = N/S = 64 and C_j = S × (64 − j). Raw IQ is not saved.
Loading the file and analyzing it gives the same summary as the analysis of the run.
The native tag is `twotone/ro_optimize/rpi_freq` with cfg schema `3.0`.

## MCP

This experiment has no MCP recipe. Use the GUI or the RPC tools.
