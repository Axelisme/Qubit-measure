**Readers:** operator, agent. Run ground-heralded single-qubit length Rabi and review pulse-duration calibration proposals.

# Heralded length Rabi

## Before Run

Bind `qubit` and `resonator` to the intended components. Supply the fixed drive frequency and gain, pulse waveform, readout generator/ADC settings and timing. A prior pi calibration is not required to scan length. The starting drive and range follow ordinary length Rabi; inspect them before Run.

The readout must contain a GE Post circle classifier calibrated in the same IQ coordinates and integration window, with ground/excited centers and radius. Linear and centroid3 classifiers are not supported. Missing classifier calibration stops acquisition. Changing readout gain, frequency, timing or integration length can invalidate the saved classifier.

Set Duration Start/Stop in us, Expts, Reps, Rounds and Relax delay. Select optional reset if needed. The hidden initialization pulse is optional, as in ordinary length Rabi. Const and flat-top waveforms use a hardware sweep. Other editable waveforms use a host scan, completing all lengths in each round. Native arb waveforms cannot change their asset-derived length and are rejected.

## Run and inspect

The sequence is optional reset → ground herald → optional initialization → length-swept drive → terminal readout. Separate herald and signal pulse-readout instances share the same settings and classifier. The host stores signal counts and IQ sums for all herald classes, g/e/other. Live plots and fits use only the herald-g conditional mean IQ. This is post-selection, not active reset or joint heralding.

Run displays the selected mean IQ projected onto a real trace. Reps counts attempted shots per length and round, not accepted shots. Across rounds and host segments, count and IQ sums add directly. Herald-g conditional means are calculated from the accumulated sums. Early-stop SNR is checked after a complete curve. Acquire retry retains the ordinary length-Rabi retry policy.

Pass fraction is the herald-g count divided by the total class count on the actual length grid. It is computed for analysis or live presentation, not stored. A completed point with no accepted shots has fraction zero and NaN IQ. Unmeasured points remain NaN. A small fraction means fewer shots contributed to that point. The classification pass fraction is not an independently measured ground purity; misclassification and transitions during readout can change it.

## Analyze and review writeback

Open Analysis and choose Fit decay envelope, `decay`, and Fit phase offset, `fit_phase`, then Analyze. Analyze takes the herald-g conditional mean IQ, removes points with nonfinite length or IQ, rotates IQ onto its signal axis and takes the real part. This experiment calls `fit_rabi`. Fit decay envelope enables a decaying oscillation envelope; disabling it fits a nondecaying oscillation. Fit phase offset allows free phase; disabling it fixes phase at 0°. Pass fraction is a diagnostic, not an extra fit weight. Analysis does not require saved cfg. All-NaN data raise `ValueError("All data are NaN!")`; other invalid or insufficient traces propagate errors.

This experiment draws the `fit` plot with rotated IQ, the fitted curve, pi/pi2 markers and uncertainty spans. The title shows the fitted oscillation frequency and its uncertainty in MHz.

Analyze returns `HeraldingLenRabiAnalysis`:

- `pi_len` and `pi_len_err` give the fitted pi duration and its standard uncertainty in us.
- `pi2_len` and `pi2_len_err` give the fitted pi/2 duration and its standard uncertainty in us.
- `rabi_f` and `rabi_f_err` give the fitted oscillation frequency and its standard uncertainty in MHz.
- None of these six numeric fields is None. `fit_quality["fit"]` contains dimensionless residual and relative-error diagnostics. The type allows `fit_quality=None` when diagnostics are not supplied; this experiment's Analyze always supplies them on success.

Inspect the oscillation, fitted pi/pi2 markers, uncertainties and fit quality. Writeback proposes the same targets as ordinary length Rabi, with every description marked Heralded:

- `$qubit.ext.rabi_freq` in MHz.
- `$qubit.pulses.pi_len` and `$qubit.pulses.pi2_len`, changing waveform length to fitted pi and pi/2 lengths. The pi_amp/pi2_amp gain-calibration pulses remain unchanged.
- `$qubit.module.x180` and `$qubit.module.x90`, pointing to those resolved pulse keys.
- `$qubit.ext.pi_product`, the source gain multiplied by the fitted pi length in gain·us.

Review the captured destination and each proposal before Apply Selected. Run and Analyze do not apply proposals. Without a saved cfg, only the frequency proposal is available. Missing captured parameters or a nonscalar gain prevents the corresponding writeback path.

## Save and load

Save All preserves `sums` with shape (Nl, 3, 3), acquisition cfg including classifier, metadata and completed analysis. The final axes are herald_states (g/e/other) and components (count/sum I/sum Q). Old files without `sums` fail with a missing-variable error. Native Load Data restores sums and the original cfg; it does not read a current classifier to replace the saved one. Analyze again on the loaded record to inspect the same finite data. Loading a saved result does not refill the current input Form.

## MCP recipe

`heralding_len_rabi` opens this experiment, selects readout and drive sources, applies fixed frequency/gain, length stop, points, reps and rounds, then applies overrides and checks readiness. `points` maps to sweep Expts. Omitted fields retain Form or analysis values; explicit null is sent to the owner. The recipe runs, analyzes, asks before writeback, and saves completed Run data even if analysis fails. The analysis controls are `decay` and `fit_phase`.
