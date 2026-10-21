# Fast flux two-tone spectroscopy

**Readers:** operator, agent. Map qubit probe frequency against fast-flux pulse gain from the GUI or MCP and inspect spectral features.

**Last updated:** 2026-10-21

## What this measures

Open `fastflux/twotone` to map qubit probe frequency against flux-pulse gain.
After reset, the flux pulse and qubit probe start together.
Readout begins after both branches finish.
The result is mean complex IQ with shape (gain, frequency).
The live map and Analyze map project IQ onto a real axis.
They locate spectral features. Transition fitting and flux calibration require
separate work.

## Before Run

Bind `qubit` and `resonator` to the intended components before Open.
Reset captures their current values again.
The probe starts from `$qubit.freq` and `$qubit.wiring.ch`.
The frequency sweep initially spans three linewidths around `$qubit.freq`.
Readout uses `$qubit.module.readout` when available, otherwise the readout template.
Reset uses `$qubit.module.reset`.

The Flux pulse module slot uses the `flux_pulse` template.
Its channel comes from `$qubit.wiring.lo_flux_ch`.
A missing or invalid channel leaves the form Invalid until you supply a confirmed channel.
The template starts with const waveform, gain 1.0, frequency 0 MHz and Nyquist zone 2.
This form starts its waveform duration at 1 us, matching the initial probe duration.
These values are editing seeds. Establish the flux calibration separately.
Set the flux pulse frequency, Nyquist zone, waveform, duration and timing for your wiring.

**Run replaces the flux pulse gain with the gain sweep.**
The initial sweep is one point at zero gain.
Set its gain range and point count explicitly before mapping.
The sweep determines the acquired gain, overriding the template's gain of 1.0.
Confirm allowed gain limits, channels, timing, reset, readout and repetition settings before Run.
Gain is normalized amplitude. Converting it to magnetic flux requires a separate
calibration. Frequency is MHz; durations are us.
Use Preview cfg to review the complete resolved configuration.

## Run and analysis

Run acquires the gain-by-frequency map while the DC flux device stays in place.
Stop retains partial data; unmeasured points remain NaN.
Analyze redraws the projected map from the selected Run or Load source.
Analysis produces a map only; calibration parameters and modules retain their values.
Review the map yourself before choosing any later calibration.

Save and Load retain frequency and gain axes and complex IQ.
Frequency is stored in Hz and restored in MHz; gain is unchanged.
Load keeps the current form unchanged and uses the saved source for historical data.
