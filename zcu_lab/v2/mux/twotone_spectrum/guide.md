**Readers:** operator, agent. Find transition centers and widths with simultaneous qubit spectroscopy.

# Mux two-tone spectrum

Sweep the qubit drives and detect spectral features in mean readout IQ. The fit estimates transition center and FWHM in MHz. It does not by itself identify the transition.

**Sequence:** common reset → simultaneous drives at one frequency per tone → fixed-frequency mux readout after the longest drive.

## Key settings

Bind the `mux` alias to a `group/mux` with matching qubits and resonators in readout order. Check the full **Readout** module, including frequencies and ADCs for inactive tones. If there is no readout slot, the inline starting values use group wiring and resonator calibrations; verify gain and timing before Run.

**Tones** chooses what to edit. **Participants** takes a nonempty list of unique zero-based positions in run order, e.g. `[2, 0]`. It selects both drives and emitted readout tones. The `participants` alias seeds those positions; if absent, unbound, or empty, the form starts with all tones. Editing the selection preserves tone values. Fill required pulse and sweep fields for any newly selected tone, using distinct drive generators separate from readout.

Set each **Tones → Frequency → Start / Stop** around the expected transition in MHz. The initial bounds are `qubit.freq ± 1.5 × qubit.kappa`; widen if the baseline or both sides of the feature are missing. Use a weak probe to limit power broadening. **Expts** gives one common count, not a grid of frequency combinations. Drives advance together; the plotted RF values are the hardware-rounded coordinates. **Model type** selects Lorentzian (`lor`) or sinc (`sinc`); each fit needs six valid points.

In `mux_twotone_spectrum`, omitted inputs retain settings. Null `participants`, `sweeps` (clears all bounds) or `n` leaves missing required values at the pre-Run readiness check unless an override fills them. Null `model_type` is rejected by Analyze, after acquisition. See the [mux recipe inputs](../../../recipes/mux-guide.md#mux_twotone_spectrum).

## Read the plots and use the result

Live `measurement_i` and Analyze `fit_i` use the original tone index `i`, not the participant-list row. Live shows background-subtracted mean IQ magnitude against drive RF. Lorentzian Analyze uses the same magnitude. Sinc Analyze uses a signed principal-axis projection of background-subtracted IQ to retain its sidelobes. Its y-axis reads **Background-subtracted IQ projection (a.u.)**. With only one finite point, it uses I; subtracting that point as the background leaves zero contrast. Check that the center marks a resolved feature and that the linewidth is larger than the spacing and inside the span. There is no pointwise shot SEM. R² measures explained variation, normalized residual RMS measures mismatch relative to signal range, and the reported ± is fit uncertainty.

Writeback proposes qubit `freq`, `kappa` and a frequency sample at the captured bias. Confirm the feature belongs to the intended transition before using the new center for Rabi. For shared controls see [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md).
