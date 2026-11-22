**Readers:** operator, agent. Measure a resonator's frequency and linewidth from a narrow one-tone RF sweep.

# One-tone resonator spectrum

Each shot applies a readout pulse and integrates the demodulated signal over the ADC window. The generator and ADC use the swept readout RF frequency. Run averages the complex I and Q at each frequency. Analyze fits both quadratures to estimate the resonator frequency and the full linewidth (FWHM). There is no qubit drive or state classification in this measurement.

## Set up the sweep

1. Bind `resonator` to the resonator under test. Use lookback to measure the flight time. Use [range find](../range_find/guide.md) first if the resonance center is unknown.
2. Check **Readout**. The inline `readout` template uses a constant envelope and a gain of 0.05 a.u. Gain is a fraction of DAC full-scale amplitude, not power in watts. **Gen ch** selects the DAC output that plays the pulse. **RO ch** selects the ADC input that receives it. Both start from the bound resonator's wiring. The frequency sweep supplies the RF frequency, so the module's fixed frequency is hidden. The ordinary one-tone form does not use a qubit reset pulse. Leave **Classifier** at None for this averaged IQ measurement.
   - **NQZ** selects the generator's output Nyquist zone (1 or 2). Use the calibrated zone for the selected output route. **Mixer freq** sets a separate mixer frequency in MHz; leave it blank on a route without one.
   - **Phase** rotates the transmitted readout tone relative to the generator's phase reference, in degrees. It does not set the Homophasal circle phase offset.
   - **Pre-delay** delays the transmitted pulse from the shot's readout start. **Trig offset** times the ADC trigger from that same start; increasing pre-delay does not move the trigger. If you move the pulse by 0.1 µs and want to keep the window at the same place in the pulse envelope, move the trigger by the same 0.1 µs.
   - **Post-delay** waits after the pulse so the cavity can empty. Start at `5/(2π·κ)` µs when κ is the linewidth in MHz; the default post-delay can be zero.
3. Check the trigger and window in µs. Ring-up time is `1/(2π·κ)` µs. The trigger offset starts at the measured flight time plus five ring-up times, or at flight time alone when κ is absent. The ADC window starts at 0.97 µs, 0.03 µs shorter than the 1.0 µs template pulse. The form extends the pulse to at least the trigger offset plus the window. If flight time is missing, supply the required trigger and pulse length rather than assuming either. Use a window within the arriving readout envelope.
4. **Readout RF frequency** starts at the captured resonator frequency ± 1.5 linewidths, with 301 points in `linrange`. Enter frequency values in MHz. Start, Stop and Center are absolute RF frequencies, not mixer offsets.
   - `linrange`: enter Start, Stop and Expts, the number of points.
   - `center`: enter Center, Span and Expts. Span is the full width, so the requested endpoints are Center − Span/2 and Center + Span/2.
   - `linrange_step`: enter Start, Stop and a fixed Step instead of a point count.
   - `center_step`: enter Center, full Span and a fixed Step instead of a point count.

   Expts modes include both endpoints. Step modes keep the first point and take as many whole steps toward the requested endpoint as fit. For a nonzero interval, Step must point in the sweep direction and cannot be larger than the interval in magnitude. The last point reaches the requested endpoint only if the interval is a whole number of steps, within rounding tolerance. For example, a 0.3 MHz Step over a 1 MHz interval ends 0.9 MHz from the first point.

   **Sampling mode** is separate and starts at Linear. With Homophasal, the same effective endpoints and point count set the grid, but the points are equally spaced in resonator-circle phase, not in RF frequency. Homophasal starts from the saved fitted resonance frequency (MHz), linewidth (MHz) and circle phase offset (rad). Fill any blank calibration fields from a previous fit. Switch back to Linear for a first fit or if the prior center is no longer reliable.
5. **Reps** starts at 1000 shots per frequency in each round. **Rounds** starts at 10 repetitions of the whole scan for averaging. **Relax delay** starts at 1 µs and sets the wait after each shot, separate from the readout pulse's post-delay. Increase Reps or Rounds when the response is noisy; independent noise falls approximately as the inverse square root of their product.

Open or Defaults captures the selected point's sources and builds editable initial values. Defaults can replace unsaved values. Refresh updates available sources and choices without replacing authored input values.

## Read the plots

Run plots the magnitude of the mean complex IQ, `|mean(I + iQ)|`, in a.u. against the acquired absolute RF grid in MHz. A round measures the entire frequency grid; later rounds average more shots at those same frequencies. This is an averaged receiver signal, not a state probability. The experiment locates a spectral feature, so it does not draw per-point shot SEM bars.

On Analyze, choose **Resonance model**: `hm` for a hanger notch, `t` for a transmission peak or `auto` to select from the magnitude trace. **Fit amplitude background slope** allows a frequency-dependent exponential amplitude envelope and starts on. **Fit phase background curvature** allows a quadratic phase term and starts off. Turn a background term off if the fit cannot separate it from the resonance over the scanned window. Compare both flanks and the IQ diagnostics before accepting the simpler fit. **Electrical delay seed** in µs selects a starting delay branch, not a fixed fit value. Leave it blank to use a saved calibration for the same generator/ADC pair, or to let Analyze search branches within 100 µs when there is no matching calibration. Use a finite signed seed, including zero, only when you have a plausible electrical delay for that route from a calibration or a prior spectrum fit. The pulse flight time just locates the ADC window and is not by itself evidence for the delay branch.

The main Analyze plot repeats the Run magnitude samples. It overlays the selected model evaluated on a dense grid and marks the fitted resonance center with a vertical line. The fit omits the first and last samples. The visual minimum of a hanger notch may differ from the fitted center for an asymmetric line. Read the frequency and linewidth in MHz from the annotation. The fitter does not return covariance, so neither number has an estimated fit standard error.

R² compares the complex fit to predicting the mean of the fit observations; 1 is exact, 0 matches the mean predictor, and negative is worse. Normalized residual RMS divides the root-mean-square residual by the observed range. Both use the interior points' I and Q samples, not just the magnitude in the first figure. The diagnostics plot shows unwrapped phase and the raw IQ locus against the same fitted response. Look for coherent disagreement there even when the magnitude curve looks close. These diagnostics have no automatic acceptance threshold.

If a flank is cut off, widen the sweep to include both sides of the resonance. A useful next window is the fitted center ± 1.5 fitted linewidths; keep at least several samples across one linewidth. For example, with a captured 7190.00 MHz center, a fitted center of 7190.12 MHz and a 1.10 MHz linewidth, raise the center by 0.12 MHz and sweep 7188.47–7191.77 MHz. If the fit misses the feature or the data do not distinguish a model, fix the window, gain or averaging and repeat before applying calibration. A large magnitude trace alone is not evidence of state discrimination.

## Review calibration and save

The Analyze summary identifies the Run or Load source and lists its configured Reps × Rounds. It does not report the number of completed shots. Load uses the saved cfg if available; it does not replace missing settings with the current form.

Analyze proposes the fitted frequency, linewidth and circle phase offset. Compare them with the Preview values before applying. A readout module proposal keeps the acquired pulse and timing settings and changes only its RF frequency. Apply the module and the following readout slot row together to select it for later qubit measurements. The slot belongs to the resonator. The delay proposal also records the generator and ADC channels. Delay can be written back only with a known route and either a trusted delay seed or a nonuniform fitting grid; a uniformly sampled trace alone can leave the absolute delay branch ambiguous.

Run uses its captured parameter source. A native file from the opened entry can use the file's historical point and role assignments for writeback, not the tab's current binding. Load without a compatible source can still fit the IQ. It does not recreate an old unsaved draft. Review the destination before applying any proposal.

## MCP recipe

`onetone_spectrum` sets Readout RF frequency › Start and Stop to `center_mhz` minus and plus half of `span_mhz` when both are numbers. Omit both to keep the endpoints. If you supply just one, even as `null`, the recipe returns `needs_parameters` naming the missing partner before acquisition. If you supply both and either is `null`, the recipe clears both Start and Stop. Unless `overrides` fills them in, the missing endpoints prevent acquisition.

`points` sets Expts. `readout_ref` selects Readout, and `gain` changes Readout › Gain. `reps` and `rounds` set the shot and averaging counts. The four Analyze inputs set the controls described above; an omitted value stays unchanged and an explicit null goes to the owning form. The recipe asks before writeback and saves the data and analysis. Its frequency and linewidth are the quantities shown in the Analyze annotation.
