**Readers:** operator, agent. Configure a single-carrier two-photon g/f scan and inspect acquired IQ.

# Single-tone Raman g/f scan

## Before opening the tab

Measure the 0–3 transition from ground with spectroscopy. Prepare level 3 with
the calibrated Pi03 pulse, then measure the 3–2 transition.
Compute the carrier center as (measured f03 − measured f32) / 2 in MHz.
The form leaves this field blank. It does not read a simulator spectrum.

Prepare a sampled arbitrary waveform asset before running. A baseline envelope
is sqrt(sin(pi*t/T)) for 0 ≤ t ≤ T, equivalent to a centered sqrt(cos) pulse.
For example, a 0.1 us asset uses the formula
`sqrt(abs(sin(pi*t/0.1)))` over a single 0.1 us segment with peak normalization.
Choose that asset in the Raman drive waveform editor.
This scan does not generate a second carrier or a pair of Lambda pulses.

Calibrate the readout using singleshot/qutrit_population at the same readout
settings. The final population analysis requires the saved readout module's
three-center g/e/f classifier and confusion matrix.
The current raw-IQ adapter has no population Analysis stage yet.
Raw IQ is not a transfer probability or an experimental gate fidelity.

## Configure the form

Open **New Tab → twotone → raman_gf**.
Choose the Raman drive generator, NQZ, phase and sampled waveform.
The scan supplies the drive frequency and gain; they are not independent fields.
Select the readout module, then confirm its generator, ADC, frequency, gain,
waveform, integration length and trigger offset.

The form lists drive and readout first, then carrier center, scan mode, ranges,
shots and relaxation time.
A kickstart entry supplies wiring and nominal readout candidates, not pulse or
readout timing calibration. Blank carrier, waveform asset or required timing
keeps Run disabled.
Entering a spectroscopy carrier alone does not complete the readout.
An existing readout slot is used on Open or Reset; its presence does not prove
that its settings or classifier remain calibrated.

The modes use the following ranges:

| Mode | Varied quantities | Held value |
| --- | --- | --- |
| frequency | Two-photon detuning | Gain start |
| area | Drive gain | Detuning start |
| frequency_area | Detuning × gain | None |

Two-photon detuning is `2*f_drive − 2*carrier_center`, in MHz.
A positive value raises the single carrier by half that detuning.
Area is `gain² * integral(abs(envelope)² dt)`, in gain² us.
It describes the source envelope, not a calibrated rotation angle.
QICK quantizes the carrier and gain; saved coordinates use those actual values.
The intensity integral uses source samples and does not account for DAC
interpolation or timing quantization.

Confirm nonnegative gain, shots per point and a relaxation interval appropriate
for ground preparation. Each point sends one Raman pulse followed immediately
by readout. There is no reset or same-run reference preparation.

## Run and save

Run acquires mean-sample ADC IQ with shape (points, shots).
Points are ordered by detuning first, then gain.
The measurement plot shows mean I and Q versus point.
Stop retains completed rows and leaves unfinished rows as NaN.
Save keeps raw IQ, actual detuning, gain and area together with this run's cfg.
It does not copy a later library change into an existing result.

After changing source parameters or library bindings, Reset captures them again
and may replace unsaved input. Refresh updates source choices but preserves
authored input and previously resolved defaults.
Reload experiments rebuilds the adapter and restores the existing tab.
Restart the application after changing a fixed framework dependency.

## Interpretation and limits

Use acquired g/e/f readout classification to distinguish g/f response from
e leakage. A circle classifier's third class is Other, not f.
Changing readout settings requires a new calibration; a stored classifier does
not automatically become invalid.

Strong-drive simulation requires full-drive non-RWA convergence checks.
A successful acquisition alone does not establish solver accuracy.
This adapter neither selects the evolution kernel nor changes solver settings.
No numerical acceptance range or gate fidelity is claimed here.
