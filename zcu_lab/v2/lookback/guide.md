**Readers:** operator, agent. Measure the arrival time of a readout pulse and use it to place the ADC window.

# Lookback

For opening, refreshing, applying and saving, see [`zcu_lab/v2/operator-workflow.md`](../operator-workflow.md).

Lookback measures the time-resolved, demodulated ADC waveform. Each shot plays one readout pulse while capturing the full ADC window, then waits before repeating. There is no qubit drive, reset, or classification. Reps is fixed at 1; each averaging round contributes one full ADC trace.

## Place the capture window

- Bind the qubit and verify its resonator and generator/ADC wiring. Check Readout frequency and gain before Run, especially the uncalibrated Inline gain of 1.0. Inline starts at resonator frequency plus linewidth, or plus 1 MHz if the linewidth is missing.
- **Trig offset** starts at stored flight time − 0.100 µs, or 0 µs when unknown. An invalid stored time leaves it blank. Open the ADC early enough to see a quiet baseline and extend **RO length** through the falling edge.
- Keep Readout **Pre-delay** at zero when calibrating flight time. Time is measured from the block start, so a programmed Pre-delay shifts the observed edge.
- Increase **Averaging rounds** if noise obscures the edge. If repeat traces carry field from the previous pulse, increase the inter-shot wait. When setting Readout `post_delay` for cavity ring-down, start at `5/(2π·κ)` µs for linewidth κ in MHz and compare the trace with a longer wait.

## Read the plots

- **Run magnitude** is `|mean IQ|` in a.u. versus time from the readout block start. The first sample is at Trig offset.
- **Analyze magnitude** overlays that trace and the magnitude after Gaussian smoothing of I and Q. The dotted threshold is the chosen fraction of the filtered peak. The arrival diamond marks the last sample strictly below threshold before the peak. If none exists, it marks **Window start fallback**, not a measured arrival.
- **Analyze quadratures** display signed I and Q of the unsmoothed average. A change in I and Q at roughly constant magnitude can indicate phase motion. If the signed samples clip, reduce Readout gain and repeat before accepting the estimate.

Adjust **Arrival threshold / peak** above the pre-pulse noise. Raising it moves a clean rising-edge estimate later. Increase **Gaussian smoothing width** only as needed: smoothing can broaden the edge and shift the estimate. Turning off **Show arrival markers** hides the marks without changing the estimate or writeback. There is no fitted model, fit error, or shot SEM.

## Use the arrival

Adopt **Estimated readout arrival** only with a clear baseline and edge on an unclipped trace. Writeback proposes the flight time for the source resonator, not a readout module edit. The ordinary readout template then uses `trigger offset = flight time + 0.015 µs` and `RO length = pulse length − 0.030 µs`.

To repeat Lookback, place Trig offset about 0.100 µs before a reliable estimate. For a fallback, move Trig offset earlier by 0.100 µs and extend RO length by the same amount to keep the window end fixed. Extend the end separately if it cuts off the pulse.

In the `lookback` recipe, `readout_ref: null` fails at the required selector before other Run edits.
Any supplied `frequency_mhz`, `readout_length_us` or `trigger_offset_us`, including `null`, edits the selected Readout as Inline without changing its library source. Omit these inputs or `rounds` to keep their values. Explicit `null` clears the corresponding required value; readiness stops before acquisition if overrides leave it blank.
If the recipe rejects an Analyze option, there is no arrival estimate or writeback question. Correct the options and Analyze the completed Run without acquiring it again. Use `plot_fit: false` to hide markers.
