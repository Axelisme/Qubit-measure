**Readers:** operator, agent. Run ground-heralded single-qubit length Rabi and review pulse-duration calibration proposals.

# Heralded length Rabi

## Before Run

Bind `qubit` and `resonator` to the intended components. Supply the fixed drive frequency and gain, pulse waveform, readout generator/ADC settings and timing. A prior pi calibration is not required to scan length. The starting drive and range follow ordinary length Rabi; inspect them before Run.

The readout must contain a classifier calibrated in the same IQ coordinates and integration window. A linear classifier needs angle and threshold. A circle classifier needs ground/excited centers and the GE Post radius. Missing classifier calibration stops acquisition. Changing readout gain, frequency, timing or integration length can invalidate the saved classifier.

Set Duration Start/Stop in us, Expts, Reps, Rounds and Relax delay. Select optional reset if needed. The hidden initialization pulse is optional, as in ordinary length Rabi. Const and flat-top waveforms use a hardware sweep. Other editable waveforms use a host scan, completing all lengths in each round. Native arb waveforms cannot change their asset-derived length and are rejected.

## Run and inspect

The sequence is optional reset → ground herald → optional initialization → length-swept drive → terminal readout. Separate herald and signal pulse-readout instances share the same settings and classifier. The host retains only signal shots accepted by the herald classifier. This is post-selection, not active reset or joint heralding.

Run displays the selected mean IQ projected onto a real trace. Reps counts attempted shots per length and round, not accepted shots. Across rounds, means are weighted by accepted counts. For host scans, zero-pass rounds contribute no weighted signal. Early-stop SNR is checked after a complete curve. Acquire retry retains the ordinary length-Rabi retry policy.

`pass_fraction` is accepted shots divided by attempted shots on the same actual length grid. A completed point with no accepted shots has fraction zero and NaN IQ. Unmeasured points remain NaN. A small fraction means fewer shots contributed to that point. The classification pass fraction is not an independently measured ground purity; misclassification and transitions during readout can change it.

## Analyze and review writeback

Open Analysis and choose Fit decay envelope and Fit phase offset, then Analyze. Nonfinite lengths and complex IQ are removed before the ordinary length-Rabi fit. Pass fraction is a diagnostic, not an extra fit weight. Too few finite points or a failed fit does not supply a usable calibration.

Inspect the oscillation, fitted pi/pi2 markers, uncertainties and fit quality. Writeback proposes the same targets as ordinary length Rabi, with every description marked Heralded:

- `$qubit.ext.rabi_freq` in MHz.
- `$qubit.pulses.pi_amp` and `$qubit.pulses.pi2_amp`, changing waveform length to fitted pi and pi/2 lengths.
- `$qubit.module.x180` and `$qubit.module.x90`, pointing to those resolved pulse keys.
- `$qubit.ext.pi_product`, the source gain multiplied by the fitted pi length in gain·us.

Review the captured destination and each proposal before Apply Selected. Run and Analyze do not apply proposals. Without a saved cfg, only the frequency proposal is available. Missing captured parameters or a nonscalar gain prevents the corresponding writeback path.

## Save and load

Save All preserves selected complex IQ, same-grid pass fraction, acquisition cfg including classifier, metadata and completed analysis. Native Load Data restores both variables and the original cfg; it does not read a current classifier to replace the saved one. Analyze again on the loaded record to inspect the same finite data. Loading a saved result does not refill the current input Form.

## MCP recipe

`heralding_len_rabi` opens this experiment, selects readout and drive sources, applies fixed frequency/gain, length stop, points, reps and rounds, then applies overrides and checks readiness. `points` maps to sweep Expts. Omitted fields retain Form or analysis values; explicit null is sent to the owner. The recipe runs, analyzes, asks before writeback, and saves completed Run data even if analysis fails. The analysis controls are `decay` and `fit_phase`.
