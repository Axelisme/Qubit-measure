## Behavior

**Readers:** operator, agent. Two-tone qubit spectroscopy: drives a qubit probe tone while reading out the resonator, sweeps the drive frequency, and fits the qubit response (Lorentzian or sinc) to extract the qubit transition frequency and its linewidth. Runs on real hardware. In first bring-up, run it after onetone/flux_dep has found `$qubit.flux_int` and onetone/freq has re-calibrated the resonator at that flux.

**Last updated:** 2026-10-21

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Parameter sources

Open and Reset fill editable starting values from captured calibration. The frequency sweep starts with 301 points at `$qubit.freq` ± 20 * `$qubit.kappa`, in MHz. The endpoints and point count remain editable. Existing edits persist until Reset evaluates the starting values again. The fresh probe reads `$qubit.freq` and `$qubit.wiring.ch`; its gain is 0.1 and waveform length is 1.0 us, independent of the calibrated x180 pulse length. The readout template reads `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch` for its tones and channels, and `$resonator.time_of_fly` + 0.05 us for trigger timing. Missing or invalid required parameter sources leave the affected fields empty and Invalid. Supply valid calibration or direct values before Run. For a first qubit search, supply the frequency window yourself. Review the hardware-safe passband and enter a user-approved wide sweep.

## Module slots

The drive starts with a fresh 'qub_probe' constant pulse for spectroscopy. Readout uses the `$qubit.module.readout` slot captured at Open/Reset; an unset slot, missing module or unbound alias uses a fresh 'readout' template with a constant waveform. That fallback still needs its required parameter sources. Reset uses the optional `$qubit.module.reset` slot; an unset slot, missing module or unbound alias leaves it disabled and Valid. Init Pulse stays disabled.

## Typical writeback

Proposes fitted qubit.freq and qubit.kappa (FWHM) in MHz. An independent freq_mhz/freq_err_mhz sample uses the captured Run bias when device evidence is available. Module slots stay unchanged. Apply only after the agent/user verifies that the plotted feature is the intended qubit line.

## MCP recipe

`twotone_spectrum` requires explicit `center_mhz`, `span_mhz` and `points`.
Supply numeric values for center and span in MHz. Span must be positive, and points
must be at least 2. The recipe writes numeric frequency endpoints from the
supplied center and span.

## Recommended

Standard flow at a fresh `$qubit.flux_int`: first run a wide survey over the user-approved passband to find a real qubit feature, and use the measured response as the evidence. Predictor output and simulator truth are not evidence of a measured qubit line. Then run a narrower scan around the observed feature and fit it. Analysis defaults to the Lorentzian fit ('lor') with plot-fit on; switch to 'sinc' for power-broadened or saturated lines. If the scan looks like noise, widen/re-centre and check readout/gain before concluding there is no qubit.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Supply an approved qubit-frequency window. Check drive gain/length and the readout frequency/timing.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, Reset to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, Reset and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `twotone_spectrum` after inspecting the same Form errors.
   center_mhz, span_mhz and points are required. points maps to frequency Expts. model_type selects lor/sinc; plot_fit controls the existing fit overlay.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
