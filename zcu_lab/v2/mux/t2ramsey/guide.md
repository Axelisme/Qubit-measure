**Readers:** operator, agent. Read mux Ramsey coherence and review frequency corrections.

# Mux Ramsey

Measure free-evolution coherence from mean IQ as the time between π/2 pulses changes. The decay envelope gives T2*; the fitted fringe rate includes the applied phase ramp and the drive-to-transition mismatch.

**Sequence:** common reset → simultaneous π/2 pulses → common delay → simultaneous phase-ramped π/2 pulses → mux readout. The delay starts after the longest first branch; shorter branches can accrue an extra fixed wait and phase.

## Key settings

Bind the `mux` alias to a `group/mux` with matching qubits and resonators in readout order. Check the full **Readout** module, including frequencies and ADCs for inactive tones. If there is no readout slot, the inline starting values use group wiring and resonator calibrations; verify gain and timing before Run.

**Tones** chooses what to edit. **Participants** takes a nonempty list of unique zero-based positions in run order, e.g. `[2, 0]`. It selects both drives and emitted readout tones. The `participants` alias seeds those positions; if absent, unbound, or empty, the form starts with all tones. Editing the selection preserves tone values. Fill required pulse and sweep fields for any newly selected tone, using distinct drive generators separate from readout.

Choose each tone's calibrated **π/2 pulse** and set **Applied phase-ramp detune** (MHz). The final pulse adds `360° × detune × delay` to the source phase; the RF carrier does not change. Choose a shared **Delay** window of about 3 × expected T2* with at least 10 points per fringe, based on the fastest expected rate. A short window shows only part of the decay, so T2* relies on extrapolation; sparse points can hide or alias fringes.

**Fit fringe** uses a damped cosine (at least six finite points); off uses an exponential (at least four). **Fit phase offset (fringe only)** lets the fit find a constant offset; when off, it uses the measured signal at the delay closest to zero to choose 0° or 180°.

In `mux_t2ramsey`, omission retains settings. Null `participants` leaves a missing selection at the pre-Run readiness check unless an override fills it. Null `fit_fringe` or `fit_phase` is rejected by Analyze, after acquisition. See the [mux recipe inputs](../../../recipes/mux-guide.md#mux_t2ramsey).

## Measurement and analysis

Live `measurement_i` uses the original tone index `i`, not the participant-list row. Read its progress in acquired delay points and accumulated shots per point. The connected markers show the signed principal-axis projection of mean IQ versus requested free-evolution delay. The projection is in a.u., not a population. With fewer than two finite points, the plot uses the I axis.

Analyze `fit_i` keeps the same points and overlays the fitted model as a solid line. The dotted line marks T2* when it falls within the fitted window. If the fit fails, the plot connects the points and reports why no model is available.

Shot SEM bars show the expected error of each projected mean: the sample standard deviation of the same shots that gave that point, divided by the square root of their count. They are not single-shot scatter, round-to-round scatter or fit uncertainty. Fewer than two shots give no bar. After reloading a data file, rerunning Analyze gives no bars because only the mean was saved; saved figures still show their original bars.

The ± on T2* and fringe rate is the fitted parameter's standard error. R² measures explained variation; normalized residual RMS is the residual RMS divided by the observed signal span. Systematic mismatch, unresolved fringes or cross-tone contamination can make a correction unreliable. Sampling uses `1 / (largest delay gap × max(|fitted fringe rate|, |actual detune|))`. This estimate cannot rule out unknown aliases.

## Next step

Writeback proposes `t2r`, a T2R sample at captured bias, and, with valid source and fringe inputs, a frequency correction. Read the Writeback review note for why a candidate was or was not preselected. Inspect the trace and fit before applying a proposal.

Match the experiment and Run ID in the Analyze summary. It lists the configured Reps×Rounds and each tone's **Applied detune** and **π/2 pulse source frequency**. Applied detune is the quantized ramp, not the nominal Form value. With a valid fringe fit and verified sign and phase wrap, use `new frequency = π/2 pulse source frequency + Applied detune − Fitted fringe rate` (MHz). If the fringe runs 0.002 MHz faster than the applied ramp, lower the source frequency by 0.002 MHz. An unsigned fringe alone cannot verify the sign: add a known positive increment to the ramp in a second Run and check that the resolved fringe rate increases by the same increment. If the sign is ambiguous or the actual ramp is unavailable, do not apply a frequency correction. Recalibrate pulses after changing the frequency; Apply does not change existing pulse modules. For shared controls see [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md).
