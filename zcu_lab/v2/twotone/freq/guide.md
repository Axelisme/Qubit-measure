**Readers:** operator, agent. Sweep the qubit drive RF frequency, identify a transition and fit its center and linewidth.

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Parameter sources

Open and Defaults fill editable starting values from captured calibration. **Probe RF frequency** starts with 301 points at `$qubit.freq` ± 20 * `$qubit.kappa`, in MHz. The endpoints and point count remain editable. Existing edits persist until Defaults evaluates the starting values again. The fresh probe reads `$qubit.freq` and `$qubit.wiring.ch`; its gain is 0.1 and waveform length is 1.0 µs, independent of the calibrated x180 pulse length. The readout template reads `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch` for its tones and channels, and `$resonator.time_of_fly` + 0.015 µs for trigger timing. Missing or invalid required parameter sources leave the affected fields empty and Invalid. Supply valid calibration or direct values before Run. For a first qubit search, supply the frequency window yourself. Review the hardware-safe passband and enter a user-approved wide sweep.

## Module slots

The drive starts with a fresh 'qub_probe' constant pulse for spectroscopy. Readout uses the `$resonator.module.readout` slot captured at Open/Defaults; an unset slot, missing module or unbound alias uses a fresh 'readout' template with a constant waveform. That fallback still needs its required parameter sources. Reset uses the optional `$qubit.module.reset` slot; an unset slot, missing module or unbound alias leaves it disabled and Valid. Init Pulse starts disabled (None). Enable it to select a calibrated library pulse or enter an inline preparation pulse. Run applies reset, Init Pulse, the swept Probe Pulse, then readout. Leave Init Pulse disabled for ground-state spectroscopy.

Bind `resonator: $qubit.resonator` so the readout follows the selected qubit's resonator. Existing points are not migrated: rerun onetone/freq and Apply its module and slot updates, or manually set `R1.module.readout` to an existing library module; otherwise this form uses the readout template. It does not read old qubit readout slots.

## Prepared-state spectroscopy

For integer-fluxonium spectroscopy, first leave Init Pulse disabled and search
for the 0–1 and 0–3 lines from ground. Enter the search windows explicitly;
the Form does not supply model transition frequencies. Refine the observed
lines and calibrate a π03 pulse with the existing Rabi experiments before using
it for preparation. In this Form enable Init Pulse, choose that π03 library
pulse or enter its measured frequency, channel, gain and waveform inline.
Sweep Probe Pulse to find 3–2 after that preparation. The Init Pulse frequency
stays fixed while the probe frequency follows the sweep.
Enter its calibrated frequency in MHz and its duration in µs.

Keep π01, π03 and π32 calibration records separate. Do not Apply a 0–3 or
prepared 3–2 fit to the generic qubit.freq slot as though it were 0–1.
Use the measured 0–3 and 3–2 carriers as Pump and Stokes inputs in subsequent
experiments. See the [integer-fluxonium environment](../../../simulation_envs/integer_fluxonium/README.md)
for offline settings, units and reference-only limits.

## Read the spectrum

Run shows the magnitude after subtracting the median complex IQ from each acquired spectrum. Its horizontal axis is the probe RF frequency in MHz and its vertical axis is background-subtracted IQ magnitude in a.u. A peak locates a change in the resonator response, not a qubit population. Run shows the acquired RF points. Analyze uses the same observable for a Lorentzian fit. The sinc option instead uses a signed IQ principal-axis projection so negative sidelobes remain visible. The Analyze acquisition summary gives Reps × Rounds from the selected Run or Load. Reps is the shot count per frequency per round, Rounds is the number of averaging passes and Relax delay is the wait after each shot. Defaults start at 1000, 10 and 1 µs. Leave Early-stop SNR empty to use all rounds.

The scan locates a peak, so it does not show pointwise shot SEM. The uncertainties next to **Qubit frequency** and **FWHM** are fit parameter errors in MHz, not uncertainties on each shot average. Use a narrow scan around the detected peak and resolve several samples per linewidth before using the width. For example, if five megahertz of the scan sits between samples and the line fits to a one-megahertz FWHM, use a step much smaller than one megahertz and retake the spectrum before writing back.

## Peak selection

Analysis has three controls: Model type, Expected center and Min SNR.
Model type selects `lor` or `sinc`.
Set Expected center, `expected_center_mhz`, to select the highest local maximum
near that frequency. A candidate is nearby within two median scan steps or
half its observable half-prominence width, whichever is larger.
Leave Expected center blank to select the most prominent candidate.
Analysis lists SNR-qualified candidates in `peaks`, ordered by prominence.
It does not apply a threshold relative to the full signal range, so a strong
transition elsewhere in a wide scan does not suppress a weak target.
Minimum width and minimum separation gates are fixed at zero.
Min SNR, `min_snr`, defaults to 5. Both prominence and height above the signal
median must reach this multiple of first-difference MAD noise sigma.
When a line occupies only 1–2 samples, its center frequency can still locate
the transition, but its FWHM cannot be resolved.
To measure linewidth, scan finely around that line.
Each candidate's `snr` is height above the median divided by noise sigma,
not prominence divided by sigma. These are statistical gates, not proof that
a candidate is a physical transition. Lowering `min_snr` can recover weak lines
and also admit more noise. See the fitting README for the estimator and limits.
Analysis starts with a window extending six half-widths on each side of the
selected peak. A candidate's width is the full width at half prominence, so
this radius is three times its reported width. For an isolated peak on a flat
background, that is six HWHM on each side.
Neighbouring candidate midpoints clip the window. Analysis then includes at
least the six samples nearest the selected center, even if that crosses a
midpoint. Close peaks no longer fail just because isolation leaves too few
samples. Overlapping transitions can still bias a single-peak fit.
The Lorentzian fit starts from the detected positive peak's center and width.
Its baseline may lie below the window minimum because the edges still contain
peak tails.
Analysis shades the actual fitted interval. If it spans at most half the measured scan, a second panel zooms into it. The dotted line marks the fitted center and gray triangles mark other candidates. Check that the fit follows the measured points inside the shaded interval, including the peak shoulders. Candidate SNR is the selected candidate's height above the median in noise sigma, not a goodness-of-fit score.
Candidate center and width are sampled estimates, not fitted frequency or FWHM.
`truncated=True` means the scan clips a half-prominence crossing.
Widen or recenter that scan before using its linewidth.
No SNR-qualified candidate near Expected center produces `AnalysisFailed`,
not a fit to a distant transition. The message reports the selected candidate
count and thresholds. A scan with fewer than six acquired samples also fails.

## Typical writeback

Proposes fitted qubit.freq and qubit.kappa (FWHM) in MHz. An independent freq_mhz/freq_err_mhz sample uses the captured Run bias when device evidence is available. Module slots stay unchanged. Apply only after the agent/user verifies that the plotted feature is the intended qubit line.

## MCP recipe

`twotone_spectrum` requires explicit `center_mhz`, `span_mhz` and `points`.
Supply numeric values for center and span in MHz. Span must be positive, and points
must be at least 2. The recipe writes numeric frequency endpoints from the
supplied center and span.

## Recommended

Standard flow at a freshly calibrated actual integer-flux bias: first run a wide survey over the user-approved passband to find a real qubit feature, and use the measured response as the evidence. Predictor output and simulator truth are not evidence of a measured qubit line. Then run a narrower scan around the observed feature and fit it. Analysis defaults to the Lorentzian fit ('lor'); switch to 'sinc' when a finite-duration probe produces visible signed sidelobes. If the scan looks like noise, widen/re-centre and check readout/gain before concluding there is no qubit.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Supply an approved qubit-frequency window. Check drive gain/length and the readout frequency/timing.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, choose Defaults to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, choose Defaults and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `twotone_spectrum` after inspecting the same Form errors.
   center_mhz, span_mhz and points are required. points maps to frequency Expts. model_type selects lor/sinc; analysis always draws the fitted overlay.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Defaults may replace unsaved inputs.
