## Behavior

One-tone resonator power dependence: a 2D sweep of readout power (gain) versus readout frequency, mapping how the resonator response shifts with drive strength. Used to find the punch-out / high-power transition and to pick a good low-power readout gain. Runs on real hardware; requires a SoC connection.

## Expects (parameter context)

Reads fixed tab parameters: resonator.freq and resonator.kappa seed the frequency sweep; resonator.wiring.ch/ro_ch seed the generator/ADC channels. Captured resonator.ext.default_probe_length minus 0.1 us seeds the ADC window, and resonator.ext.time_of_fly plus 0.05 us seeds trigger timing. Missing or null calibration and unbound aliases leave required fields Invalid without numeric fallback; source failures propagate.

## Expects (ModuleLibrary)

Starts with a fresh inline pulse readout and const waveform. It does not adopt a library module or named waveform.

## Typical writeback

No writeback — this adapter has no analysis step (the underlying experiment is measurement-only). It produces a 2D map for visual inspection only; read off the punch-out power and low-power frequency by eye and update parameters in another step.

## MCP recipe

`onetone_spectrum_over_power` can omit `center_mhz` and `span_mhz`. Omitted
center uses current `$resonator.freq`; omitted span uses 3 times current
`$resonator.kappa` for the full frequency span. The recipe reads these values
during preparation and writes numeric endpoints at center ± span/2, not live
formulas. Omitted `freq_points` retains the GUI count.

## Recommended

No analysis. Typical sweep: gain ~0.001 to 0.5 over ~101 points (low to high power), frequency resonator.freq ± linewidths over ~201 points. 'earlystop_snr' starts disabled (None or nonpositive): set it positive to stop a column early once a signal-to-noise target is reached, speeding up the scan. Narrow the gain range once you've located the transition.
