**Readers:** operator, agent. Run a single-carrier two-photon g/f scan and analyze acquired g/e/f counts with saved readout calibration.

# Single-tone Raman g/f scan

## Before opening the tab

Measure the 0–3 transition from ground with spectroscopy. Prepare level 3 with
the calibrated Pi03 pulse, then measure the 3–2 transition.
Compute the carrier center as (measured f03 − measured f32) / 2 in MHz.
The form leaves this field blank. It does not read a simulator spectrum.

Choose any supported drive waveform: arb, cosine, gauss, drag, const or flat_top.
The default editor selects arb and needs a waveform asset.
To use sqrt(sin(pi*t/T)) for 0 ≤ t ≤ T, select arb and prepare an asset;
this shape is equivalent to a centered sqrt(cos) pulse.
For example, a 0.1 µs asset uses the formula
`sqrt(abs(sin(pi*t/0.1)))` over a single 0.1 µs segment with peak normalization.
Choose that asset in the Raman drive waveform editor.
This scan does not generate a second carrier or a pair of Lambda pulses.
For two-generator Lambda pulses, use the separate STIRAP
[delay](../stirap/delay/guide.md), [amplitude](../stirap/amplitude/guide.md) or
[detuning](../stirap/detuning/guide.md) experiment.
Raman and STIRAP scans remain available as different pulse sequences and tuning entry points.
The Raman simulation checks predate the current drive-amplitude scale and have not been repeated, so successful STIRAP runs do not replace Raman validation.

Calibrate the readout using singleshot/qutrit_population at the same readout
settings. The final population analysis requires the saved readout module's
three-center g/e/f classifier and confusion matrix.
Apply that acquired reference calibration to the readout library before opening
or Resetting the Raman tab. Run requires the centroid3 classifier captured
in this run's cfg and rejects missing, circle and linear classifiers before acquisition.
Analysis uses the saved confusion matrix. It never substitutes the current library
or collects new references. Preparation-label fractions are not gate fidelity.

## Configure the form

Open **New Tab → twotone → raman_gf**.
Choose the Raman drive generator, NQZ, phase and waveform.
The scan supplies the drive frequency and gain; they are not independent fields.
Select the readout module, then confirm its generator, ADC, frequency, gain,
waveform, integration length and trigger offset.

The form lists drive and readout first, then carrier center, scan mode, ranges,
shots and relaxation time.
A kickstart entry supplies wiring and nominal readout candidates, not pulse or
readout timing calibration. A blank carrier or required timing keeps Run disabled.
An arb waveform also needs its asset before Run becomes available.
Entering a spectroscopy carrier alone does not complete the readout.
An existing readout slot is used on Open or Defaults; its presence does not prove
that its settings or classifier remain calibrated.

The modes use the following ranges:

| Mode | Varied quantities | Held value |
| --- | --- | --- |
| frequency | Two-photon detuning | Gain start |
| gain | Drive gain | Detuning start |
| frequency_gain | Detuning × gain | None |

Two-photon detuning is `2*f_drive − 2*carrier_center`, in MHz.
A positive value raises the single carrier by half that detuning.
Drive gain is a DAC full-scale fraction (a.u.), not a calibrated rotation angle.
QICK quantizes the carrier and gain; saved coordinates use those actual values.
The scan does not calculate a two-photon area or normalize gains between waveforms.

Drive gain may be negative, zero or positive, within [-1, 1].
Confirm shots per point and the relaxation interval for ground preparation before running.
Each point sends one Raman pulse followed immediately by readout. There is no reset or same-run reference preparation.

## Run and save

Run classifies each point's repetitions into g/e/f counts with shape (points, 3).
The shots field sets repetitions per point in one round, not a saved shot axis.
Points are ordered by detuning first, then gain.
The measurement plot shows counts divided by the total count at each point.
Stop retains completed rows and leaves unfinished rows as NaN.
Save keeps counts, actual detuning and gain together with this run's cfg.
It does not copy a later library change into an existing result.

After changing source parameters or library bindings, Defaults captures them again
and may replace unsaved input. Refresh updates source choices but preserves
authored input and previously resolved defaults.
Reload experiments rebuilds the adapter and restores the existing tab.
Restart the application after changing a fixed framework dependency.

## Analyze and restore

Open Analysis and click Analyze after Run. The measurement figure retains raw
g/e/f fractions. Analysis normalizes counts and shows raw and confusion-corrected g/e/f
estimates separately. Frequency and gain modes show two sets of three curves.
Frequency-gain mode shows six maps with actual detuning and drive-gain axes.
The title reports the saved training matrix's condition number and minimum
singular value. The summary contains only the training confusion matrix and
these two diagnostics; per-point fractions stay in the figures.
Correction solves in preparation-label coordinates and keeps
negative and above-one values. It does not clip or renormalize them.

The e curve is a leakage-label estimate. Preparation contamination remains part
of the calibration, so these estimates are not pure energy-state populations
or experimental gate fidelity. A circle classifier's third class is Other, not f.
Analysis rejects incomplete, nonfinite or empty counts and singular confusion matrices.

Save All preserves g/e/f counts and this run's readout cfg in native data.h5.
Old raw-shot files are rejected with an error naming the missing counts variable.
Data → Load Data → File… restores a new 4.0 run for Analyze without recalibration.
A later live library edit cannot change its classification. Reload experiments
discards results after confirmation and restores tab inputs, not acquired data.
A fresh process may restore inputs through the configured startup script.
Load a native file saved with cfg schema 4.0 to recover its data and cfg for Analyze.

Older 3.x files in `frequency`, `area` and `frequency_area` modes retain their
raw counts, detuning, gain, area and metadata. The current result restores only
counts, detuning and gain. The loader does not restore a typed cfg because the
schema major version differs, so these files cannot be analyzed directly.
It does not migrate the cfg, rename modes or substitute a current readout calibration.

## MCP recipe

Use `recipe_guide(adapter="twotone/raman_gf")` to read this guide.
The single `raman_gf` tool runs this adapter through Run, Analyze and Save.
`carrier_mhz` is the operator-supplied spectroscopy center, not a model value.
`drive_ref` selects a drive library module. `waveform_asset` supplies an existing
sampled arbitrary asset key; it converts the drive and waveform to inline while
preserving their family. Choose an arbitrary waveform source before setting
that key. The default Raman Form already uses the arbitrary family.
`readout_ref` selects the acquired centroid3 readout library module.

`scan_mode` accepts frequency, gain and frequency_gain.
`detuning_start_mhz`, `detuning_stop_mhz` and `detuning_n` set the detuning axis.
`gain_start`, `gain_stop` and `gain_n` set the signed native gain axis.
Each `n` maps to its saved cfg `expts`. Frequency holds gain start; gain holds
detuning start. Detuning is two-photon MHz; a positive offset raises the single
carrier by half that value. Drive gain uses DAC full-scale fraction (a.u.).
`shots` sets repetitions per point; `relax_delay_us` uses µs.

Omitted inputs retain Form values and sources; explicit null goes unchanged to
the owner. Python KEEP marks omission and is not a JSON value. Ordered
`overrides` apply after named inputs and before validation. Missing or invalid
Form inputs stop before Run and retain owner paths and reasons.

The report gives the saved confusion matrix condition number and minimum
singular value. Full raw/corrected g/e/f arrays remain in the analysis summary
and saved artifacts. There is no writeback proposal or question. Analysis
failure still saves g/e/f counts; inspect each Save output receipt. Analysis never
reads live calibration or collects fresh references. Restart the MCP server
after changing recipe declarations; GUI Reload does not reload that process.

## Interpretation and limits

Changing readout settings requires a new calibration; a stored classifier does
not automatically become invalid.

Strong-drive simulation requires full-drive non-RWA convergence checks.
A successful acquisition alone does not establish solver accuracy.
This adapter neither selects the evolution kernel nor changes solver settings.
The integer-fluxonium environment defaults to oscillator cutoff 80 and
20 retained levels. Its [field guide](../../../simulation_envs/integer_fluxonium/field-guide.md)
records an earlier 12-point Raman characterization, the reference/STIRAP
regressions and single-point acquire wall times at 80/15 and 80/20.
Those checks do not validate the current Raman drive-amplitude scale,
other drives or gate fidelity.
Existing working YAML files keep their own basis until reviewed and edited;
restart the environment after changing them.
