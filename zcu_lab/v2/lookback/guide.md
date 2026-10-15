## Behavior

Readout time-of-flight (lookback) measurement: fires the readout pulse and captures the decimated ADC trace versus time, so you can see when the readout signal actually arrives at the digitizer. Runs on real hardware (reps is locked to 1 — decimated acquisition has no multi-rep averaging). Run this first when bringing up a new readout chain or after changing cabling, to calibrate the trigger offset.

## Expects (parameter context)

Defaults use the tab's fixed resonator capture. resonator.freq seeds the pulse and ADC frequency; resonator.wiring.ch/ro_ch seed generator and ADC channels. Trigger offset is resonator.ext.time_of_fly - 0.1 us. Missing or null calibration and unbound aliases leave the affected fields required and cfg Invalid until supplied. No numeric fallback is substituted.

## Expects (ModuleLibrary)

Readout always starts as a fresh inline pulse-readout, even when a calibrated readout slot exists. Its const waveform is inline, not a named library waveform. Optional reset and init pulse start disabled; you can select their modules explicitly before Run.

## Typical writeback

Proposes the detected signal-arrival time in us at captured resonator.ext.time_of_fly. The writeback service resolves the destination on the analysis capture, not current focus. No module or waveform writeback.

## Recommended

Analysis defaults: threshold 'ratio'=0.1 (the offset is the latest time before the peak where the magnitude is still below 10% of maximum), 'smooth' sigma=1.0, plot-fit on. Use a wide readout window so the full pulse arrival is captured; raise 'smooth' if the trace is noisy and the offset jitters, lower 'ratio' if it triggers too early on baseline ripple.
