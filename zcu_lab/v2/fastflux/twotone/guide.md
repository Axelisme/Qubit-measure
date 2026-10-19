# Fast flux two-tone spectroscopy

## What this measures

Open `fastflux/twotone` to map qubit probe frequency against flux-pulse gain.
After reset, the flux pulse and qubit probe start together.
Readout begins after both branches finish.
The result is mean complex IQ with shape (gain, frequency).
The live map and Analyze map project IQ onto a real axis.
They locate spectral features, not a fitted transition or a calibrated flux.

## Before Run

Bind `qubit` and `resonator` to the intended components before Open.
Reset captures their current values again.
The probe starts from `$qubit.freq` and `$qubit.wiring.ch`.
The frequency sweep initially spans three linewidths around `$qubit.freq`.
Readout uses `$qubit.module.readout` when available, otherwise the readout template.
Reset uses `$qubit.module.reset`.

The flux pulse uses the `flux_pulse` role template.
Its channel comes from `$qubit.wiring.lo_flux_ch`.
A missing or invalid channel stays invalid; there is no fallback channel.
The role starts with const waveform, gain 1.0, frequency 0 MHz and Nyquist zone 2.
This form starts its waveform duration at 1 us, matching the initial probe duration.
These values are not a flux calibration.
Set the flux pulse frequency, Nyquist zone, waveform, duration and timing for your wiring.

**Run replaces the flux pulse gain with the gain sweep.**
The initial sweep is one point at zero gain.
Set its gain range and point count explicitly before mapping.
The role's gain of 1.0 is not the acquired gain; the sweep determines it.
Confirm allowed gain limits, channels, timing, reset, readout and repetition settings before Run.
Gain is normalized amplitude, not magnetic flux. Frequency is MHz; durations are us.
Use Preview cfg to review the complete resolved configuration.

## Run and analysis

Run acquires the gain-by-frequency map without moving a DC flux device.
Stop retains partial data; unmeasured points remain NaN.
Analyze redraws the projected map from the selected Run or Load source.
It has no fit options, numeric calibration or writeback proposal.
Review the map yourself before choosing any later calibration.

Native Save and Load retain the core's frequency and gain axes and complex IQ.
Frequency is stored in Hz and restored in MHz; gain is unchanged.
Load does not replace the current form or borrow its settings for historical data.
