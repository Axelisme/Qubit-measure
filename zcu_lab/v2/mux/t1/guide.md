**Readers:** operator, agent. Measure energy relaxation for each selected mux qubit.

# Mux T1

Measure mean IQ after a calibrated π pulse and variable wait. T1 is the exponential energy-relaxation time; a second fitted time is a model component, not evidence by itself of another state.

**Sequence:** common reset → simultaneous π pulses → common delay beginning after the longest pulse branch → mux readout.

## Key settings

Bind the `mux` alias to a `group/mux` with matching qubits and resonators in readout order. Check the full **Readout** module, including frequencies and ADCs for inactive tones. If there is no readout slot, the inline starting values use group wiring and resonator calibrations; verify gain and timing before Run.

**Tones** chooses what to edit. **Participants** takes a nonempty list of unique zero-based positions in run order, e.g. `[2, 0]`. It selects both drives and emitted readout tones. The `participants` alias seeds those positions; if absent, unbound, or empty, the form starts with all tones. Editing the selection preserves tone values. Fill required pulse and sweep fields for any newly selected tone, using distinct drive generators separate from readout.

Choose **Delay** in µs to resolve the initial change and reach a baseline. The stop starts at the shared **Relax delay**, the largest five-T1 recovery seed over the full group (150 µs per missing T1, using a 30 µs T1 seed). A short window without a tail makes T1 depend on extrapolation. Choose **Sampling** `decay` to concentrate points within the same bounds; the rounded times must stay distinct. Linear plots use requested times, decay sampling uses the quantized times.

**Skip leading points** excludes early delays before filtering nonfinite data. Use **Dual exponential** only when the trace resolves two components.

In `mux_t1`, omitted inputs retain settings. Null `participants` leaves a missing selection at the pre-Run readiness check unless an override fills it. Null `dual_exp` or `skip` is rejected by Analyze, after acquisition. See the [mux recipe inputs](../../../recipes/mux-guide.md#mux_t1).

## Read the plots and use the result

Live `measurement_i` and Analyze `fit_i` use the original tone index `i`, not the participant-list row. Live shows delay coverage after each completed round. Analyze adds the exponential and marks T1 if inside the scan. Compare the contrast, tail and baseline; the principal-axis mean IQ is not excited-state probability.

Bars in Live and Analyze are shot SEM: the projected shot sample standard deviation divided by √(shot count). They describe uncertainty of the point mean, not fit error, shot spread or round-to-round spread. Mean-only files still fit without bars, and fewer than two shots give no bar. R² measures explained variation, normalized residual RMS measures mismatch relative to signal range, and the reported ± is fit uncertainty.

Writeback proposes the qubit's `t1` with its fitted error and a T1 sample at the captured bias. Use a supported T1 estimate to choose recovery and timing for later runs. For shared controls see [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md).
