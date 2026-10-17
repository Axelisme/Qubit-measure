## Behavior

Readout time-of-flight (lookback) measurement: fires the readout pulse and captures the decimated ADC trace versus time, so you can see when the readout signal actually arrives at the digitizer. Runs on real hardware (reps is locked to 1 — decimated acquisition has no multi-rep averaging). Run this first when bringing up a new readout chain or after changing cabling, to calibrate the trigger offset.

## Expects (parameter context)

Defaults use the tab's fixed resonator capture. resonator.freq seeds the pulse and ADC frequency; resonator.wiring.ch/ro_ch seed generator and ADC channels. Missing/null time_of_fly starts trigger offset at 0 us, with an editable 1.5 us ADC window. Valid TOF is a finite non-bool real number >= 0; trigger offset is TOF - 0.1 us even when the result is negative, without clamping. Stored invalid TOF leaves trigger offset required and empty; alias/path/source failures propagate. These editable initial values are not calibration writeback. Increase the ADC window if arrival time plus pulse duration exceeds 1.5 us. Other missing required calibration remains Invalid.

## Expects (ModuleLibrary)

Readout always starts as a fresh inline pulse-readout, even when a calibrated readout slot exists. Its const waveform is inline, not a named library waveform. Reset and Init pulse are not exposed in this Form; the core retains both fields.

## Typical writeback

Proposes the detected signal-arrival time in us at captured resonator.ext.time_of_fly. The writeback service resolves the destination on the analysis capture, not current focus. No module or waveform writeback.

## Recommended

Analysis defaults: threshold 'ratio'=0.1 (the offset is the latest time before the peak where the magnitude is still below 10% of maximum), 'smooth' sigma=1.0, plot-fit on. Use a wide readout window so the full pulse arrival is captured; raise 'smooth' if the trace is noisy and the offset jitters, lower 'ratio' if it triggers too early on baseline ripple.
