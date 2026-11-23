**Readers:** operator, agent. Measure echo coherence for the selected mux qubits.

# Mux Echo

Measure the loss of refocused coherence as the total delay increases. The decay envelope gives T2E; any fitted fringe rate describes modulation of the echo trace, not a qubit frequency calibration.

**Sequence:** common reset → simultaneous π/2 → half delay → simultaneous π → half delay → phase-ramped π/2 → mux readout. Each stage waits for its longest branch, so unequal pulse lengths can change the balance of the two waits.

## Key settings

Bind the `mux` alias to a `group/mux` with matching qubits and resonators in readout order. Check the full **Readout** module, including frequencies and ADCs for inactive tones. If there is no readout slot, the inline starting values use group wiring and resonator calibrations; verify gain and timing before Run.

**Tones** chooses what to edit. **Participants** takes a nonempty list of unique zero-based positions in run order, e.g. `[2, 0]`. It selects both drives and emitted readout tones. The `participants` alias seeds those positions; if absent, unbound, or empty, the form starts with all tones. Editing the selection preserves tone values. Fill required pulse and sweep fields for any newly selected tone, using distinct drive generators separate from readout.

Use calibrated π/2 and π pulses on the same generator for each qubit. **Applied phase-ramp detune** (MHz) adds `360° × detune × total delay` to the final π/2 phase, without changing the drive carrier. Choose a **Delay** window of about 3 × expected T2E. For fringes, choose at least 10 points per cycle at the fastest expected rate. A short window shows only part of the envelope, leaving the fit to extrapolate; sparse points can hide or alias fringes.

**Fit method** defaults to decay; zero detune forces decay even with fringe selected. Fringe needs known nonzero source detune, or saved actual detune if source cfg is unavailable. **Fit phase offset (fringe only)** frees the phase, which otherwise stays at zero. Analyze drops the original first point; allow at least six remaining finite points for fringe or four for decay.

In `mux_t2echo`, omission retains settings. Null `participants` leaves a missing selection at the pre-Run readiness check unless an override fills it. Null `fit_method` or `fit_phase` is rejected by Analyze, after acquisition. See the [mux recipe inputs](../../../recipes/mux-guide.md#mux_t2echo).

## Read the plots

Live `measurement_i` and Analyze `fit_i` use the original tone index `i`, not the participant-list row. Both plot **IQ projection (a.u.)** against requested total echo delay. This is the signed projection of mean IQ onto its principal axis, not a population. Live connects acquired points and tracks points and accumulated shots per point. Analyze omits the first requested point and draws a solid model line; the dotted line marks T2 echo within the fitted window. If the fit fails, Analyze connects the remaining points and shows the reason instead of a model. Each plot uses the I axis if fewer than two finite points are available for estimating the principal axis.

Shot SEM bars show the expected error of each projected mean: the sample standard deviation of the same shots that gave that point, divided by the square root of their count. They are not single-shot scatter, round-to-round scatter or fit uncertainty. Fewer than two shots give no bar.

After loading a Native file with shot moments, run Analyze with the same options to see the same bars as before saving. Older Native files without shot moments still load; Analyze shows no bars for those files. Labber exports do not include shot moments.

The ± on T2 echo or fringe rate is the fitted parameter's standard error. R² measures explained variation; normalized residual RMS is the residual RMS divided by the observed signal span. Match the experiment and Run ID in the Analyze summary to see the Run's configured Reps×Rounds and the actual quantized applied detune, not the nominal Form detune.

### Multi-tone readout control

A Participants change alters both drives and emitted readout tones. To test a suspect second readout tone, keep participants, pulses, ADCs and timing fixed and change only its readout gain; a one-qubit Run answers a different question.

## Next step

Writeback proposes `t2e` and a T2E sample at captured bias, not frequency. Read the Writeback review note for why a candidate was or was not preselected. Check the decay and reject a contaminated estimate even if the fit converges. For shared controls see [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md).
