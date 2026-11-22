**Readers:** operator, agent. Measure the arrival time of a readout pulse and use it to place the ADC window.

# Lookback

Lookback plays a readout pulse and records the time-resolved, demodulated ADC waveform. Each shot consists of a readout pulse and its ADC capture window, followed by the inter-shot wait. The GUI and recipe do not apply a qubit drive or reset. Use Lookback when bringing up a readout path or after changing its cabling.

## Set up the capture

1. Select the qubit and check that its resonator is the one you want to probe. The captured resonator supplies the initial wiring and timing values. If you change the point or parameter source, use **Defaults** to recapture them. Refresh keeps your current field values. Defaults can replace unsaved edits.
2. Choose **Readout**. Use the inline pulse for a first capture, or select a calibrated readout module from the library and check its settings. The ADC must open before the pulse reaches it and stay open through the part of the pulse you need to see.
3. Run and look for a quiet baseline, a rising edge and a falling edge. Then Analyze. The summary identifies the source run and gives the requested Reps × Rounds. Lookback fixes Reps at 1, so ten rounds average ten full ADC traces if the run completes.

| Run field | What to set and where the default comes from |
| --- | --- |
| Readout › Waveform | Pulse envelope. Inline starts with a constant pulse of Length 1.0 µs. Use a library waveform or choose an inline envelope to change its shape. |
| Readout › Gen ch / RO ch | Generator output and ADC input channel numbers. The resonator wiring seeds both. Check the two ends of the same readout path. |
| Readout › Frequency (MHz) | Shared RF tone for the generator and ADC demodulation. Inline starts at resonator frequency plus linewidth, or plus 1 MHz if the linewidth is unknown. Supply the frequency if the resonator has no valid value. |
| Readout › Gain (a.u.) | Signed fraction of DAC full scale, from −1 to 1. Inline starts at 1.0, not a calibrated gain. Verify the readout drive strength and reduce it if the ADC clips. |
| Readout › RO length (µs) | ADC capture duration, independent of the pulse length. Inline starts at 1.5 µs. Increase it if the capture ends before the falling edge. |
| Readout › Trig offset (µs) | Time when the ADC window opens, relative to the start of the readout block. Inline opens 0.1 µs before the stored flight time, or at 0 µs if there is none. An invalid stored time leaves the field empty. |
| Inter-shot wait (µs) | Wait after a shot before repeating. Starts at 0 µs. Increase it if repeat traces show a residual field from the previous pulse. |
| Averaging rounds (count) | Number of one-shot traces to average, initially 10. Increase it if random noise obscures the edge. |

The Readout Advanced settings normally keep their defaults. **NQZ** is the generator's Nyquist zone, initially 2. **Phase (deg)** sets the carrier phase, initially 0°. **Mixer freq (MHz)** sets the generator mixer frequency when one is used. Leave it empty otherwise. **Pre-delay (µs)** waits before emitting the pulse; it moves the observed edge relative to the time origin. **Post-delay (µs)** waits after the pulse to let the cavity empty. Start with 5/(2πκ) µs if the measured linewidth κ is in MHz. Both delays initially use 0 µs. **Classifier** is optional calibration for other readout experiments. Lookback does not classify the trace.

## Interpret the plots

- **Run, Readout magnitude.** The horizontal axis is time from the programmed readout block start. The first trace sample is at Trig offset; the block starts before the generator's Pre-delay if that field is nonzero. The vertical axis is the magnitude of the complex ADC signal after averaging I and Q, in arbitrary units. Each shot covers the full window. The rounds bar below the Run figure counts completed rounds. The figure shows their running average; an exported plot does not include the bar.
- **Analyze, Readout magnitude.** The blue points reproduce the last Run trace. The dashed curve shows the magnitude after Gaussian smoothing of I and Q. The horizontal dotted line is the selected fraction of the smoothed peak. The diamond marks the **Estimated readout arrival**: the last sample before the peak with magnitude strictly below the threshold. When no such sample exists, the algorithm marks the first sample as **Window start fallback**. Do not accept a fallback as a measured arrival.
- **Analyze, Readout quadratures.** I and Q are the real and imaginary parts of the same unsmoothed mean ADC waveform, in a.u. Compare the two components if the magnitude changes: changes in I and Q with nearly constant magnitude indicate a phase change. Flat-topped signed samples may indicate clipping. These traces are not state populations.

Lookback locates an edge in time rather than comparing calibrated signal levels point by point. The result stores a mean trace, not the dispersion of individual samples across shots. It does not display shot SEM. The arrival depends on sample spacing, smoothing and threshold. There is no fitted model, fit-quality score, or statistical error attached to the arrival time.

| Analyze field | What it changes |
| --- | --- |
| Arrival threshold / peak (a.u.) | Fraction of the filtered peak, initially 0.1, or 10%. With a clean monotonically rising edge, raising the threshold moves the estimate later. A threshold that intersects the pre-pulse noise does not locate a reliable arrival. |
| Gaussian smoothing width (samples) | Gaussian standard deviation applied to complex samples before taking their magnitude, initially 1 sample. Raising it suppresses rapid noise but can also broaden the edge and shift the estimate. |
| Show arrival markers | Shows the threshold and arrival diamond when checked, initially on. Turning it off does not change the estimated time or writeback proposal. |

## Set the next window

If the baseline and rising edge are visible and stable, write back the **Estimated readout arrival** as the flight time of the resonator captured for this run. For a flight-time calibration, keep Pre-delay at 0 µs so the result does not include an intentional wait. Writeback does not edit the readout module. The normal `readout` template uses the saved flight time plus 0.015 µs for its trigger and leaves another 0.015 µs of pulse after the ADC window. For example, an estimate of 0.500 µs gives a normal trigger offset of 0.515 µs and a 0.970 µs ADC window for a 1.000 µs pulse.

If you need to repeat Lookback, keep the ADC window early instead. Start it about 0.100 µs before a reliable estimate to retain the baseline. If you only see a window-start fallback, move Trig offset earlier by 0.100 µs and extend RO length by the same amount to keep the window end fixed. If the signal is still rising or the falling edge is missing at the right edge, extend RO length until the plot contains them. Do not write back an arrival from a clipped, truncated or noise-dominated trace.

## Run from the recipe

The recipe opens a new Lookback tab unless `reuse_tab_id` names an idle Lookback tab. Omitting `reuse_tab_id` or passing `null` opens a new tab. The recipe applies Defaults in either case. Use `target` to select the qubit and its resonator for those defaults. Omit it or pass `null` to use the current session aliases.

The recipe applies `readout_ref` first to select a Readout module. Omit it to keep the selection from Defaults. The input schema accepts `readout_ref: null`, but the required Readout selector rejects the attempt to disable it before acquisition.

`frequency_mhz` sets **Frequency** in MHz, `readout_length_us` sets **RO length** in µs, and `trigger_offset_us` sets **Trig offset** in µs. If you supply any of these three inputs, even as `null`, the recipe first copies the selected library Readout into the tab as Inline, then applies the supplied field edits. If Readout is already Inline, the recipe edits it in place. Neither edit writes back to the library module. Omit all three to leave the selected source unchanged.

`rounds` sets **Averaging rounds**. Omit any of the four Run value inputs to keep its field value. A `null` leaves that required field invalid, rather than restoring its default. The recipe applies `overrides` in order after the named Run edits. If a required value is still missing, the Form check stops the recipe before acquisition. Omit `overrides` for no additional edits. An explicit `overrides: null` fails input validation before a tab opens.

After Run, the recipe applies `ratio`, `smooth`, and `plot_fit` to **Arrival threshold / peak**, **Gaussian smoothing width**, and **Show arrival markers**, respectively. Omit an input to keep its current Analyze value. The recipe schema accepts `null` for these inputs, but Analyze rejects it when validating the settings after Run and before analysis starts. Null does not turn off smoothing or markers. Set `plot_fit` to `false` to hide the markers.

When Analyze rejects an input, the recipe does not produce an arrival estimate or ask for writeback. It still attempts Save for the completed Run. Check the Run's Save report for the raw data outcome. You can correct the settings and Analyze the existing Run, or load the saved Run and Analyze it, without acquiring it again. When analysis succeeds, the recipe asks before writing the estimate to the resonator.
