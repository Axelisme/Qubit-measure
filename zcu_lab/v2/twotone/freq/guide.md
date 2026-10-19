## Behavior

Two-tone qubit spectroscopy: drives a qubit probe tone while reading out the resonator, sweeps the drive frequency, and fits the qubit response (Lorentzian or sinc) to extract the qubit transition frequency and its linewidth. Runs on real hardware. In first bring-up, run it after onetone/flux_dep has found `$qubit.flux_int` and onetone/freq has re-calibrated the resonator at that flux.

## Expects (parameter context)

Opening a tab or Reset evaluates Form starting expressions once against captured parameters into direct input. Range.around seeds 301 points at `$qubit.freq` ± 1.5 * `$qubit.kappa`, in MHz. The fresh probe reads `$qubit.freq` and `$qubit.wiring.ch`; its gain is 0.1 and waveform length is 1.0 us, independent of x180/pi_len. The inline readout fallback reads `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch` for its tones and channels, and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Missing or invalid required expression sources leave the affected fields empty and Invalid. Supply valid calibration or direct values before Run. There is no numeric frequency-survey fallback; for a first qubit search, review the hardware-safe passband and enter a user-approved wide sweep.

## Expects (ModuleLibrary)

Starts with a fresh inline 'qub_probe' const pulse, not a calibrated state-prep pulse. `$qubit.module.readout` resolves to a captured library link; an unset/dangling slot or unbound alias uses a fresh inline 'readout' template with a const waveform. That fallback still needs its required parameter sources. `$qubit.module.reset` is optional; an unset/dangling slot or unbound alias leaves Reset disabled and Valid. Init Pulse stays disabled.

## Typical writeback

Proposes fitted qubit.freq and qubit.kappa (FWHM) in MHz. An independent freq_mhz/freq_err_mhz sample uses the captured Run bias when device evidence is available. No module writeback. Apply only after the agent/user verifies that the plotted feature is the intended qubit line.

## MCP recipe

`twotone_spectrum` requires explicit `center_mhz`, `span_mhz` and `points`.
None can be null. Center and span use MHz, span must be positive, and points
must be at least 2. The recipe writes numeric frequency endpoints from the
supplied center and span.

## Recommended

Standard flow at a fresh `$qubit.flux_int`: first run a wide survey over the user-approved passband to find a real qubit feature, without using predictor output or simulator truth as the answer. Then run a narrower scan around the observed feature and fit it. Analysis defaults to the Lorentzian fit ('lor') with plot-fit on; switch to 'sinc' for power-broadened or saturated lines. If the scan looks like noise, widen/re-centre and check readout/gain before concluding there is no qubit.
