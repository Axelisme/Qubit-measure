**Readers:** operator, agent. Calibrate drive gain for π and π/2 pulses on a mux group.

# Mux amplitude Rabi

Sweep each selected drive's gain at fixed pulse length and measure the resulting oscillation in mean IQ. The cosine fit gives the π gain; π/2 gain is half that value for the same waveform and length.

**Sequence:** common reset → simultaneous drives at one gain per tone → mux readout after the longest branch.

## Key settings

Bind the `mux` alias to a `group/mux` with matching qubits and resonators in readout order. Check the full **Readout** module, including frequencies and ADCs for inactive tones. If there is no readout slot, the inline starting values use group wiring and resonator calibrations; verify gain and timing before Run.

**Tones** chooses what to edit. **Participants** takes a nonempty list of unique zero-based positions in run order, e.g. `[2, 0]`. It selects both drives and emitted readout tones. The `participants` alias seeds those positions; if absent, unbound, or empty, the form starts with all tones. Editing the selection preserves tone values. Fill required pulse and sweep fields for any newly selected tone, using distinct drive generators separate from readout.

Under **Tones → Gain**, choose **Start** and **Stop** within −1 to 1. Stop starts at `min(1.0, 1.2 × x180 gain)`; set it explicitly if the x180 calibration is missing. Sweep through the expected first inversion; **Expts** is shared, but each tone has its own gain range. Set **Skip points** only if you have a reason to exclude the first measured gains; fit needs at least six finite points afterward.

In `mux_amp_rabi`, omitted inputs retain settings. Null `participants`, `sweeps` (clears all bounds) or `n` leaves missing required values at the pre-Run readiness check unless an override fills them. Null `skip` is rejected by Analyze, after acquisition. See the [mux recipe inputs](../../../recipes/mux-guide.md#mux_amp_rabi).

## Read the plots and use the result

Live `measurement_i` and Analyze `fit_i` use the original tone index `i`, not the participant-list row. Live shows which rounded gain points were measured. Analyze adds the cosine and marks π gain if in range. Look for a resolved inversion in the range and compare the curve with the points. Both plots label the signed projection of mean IQ **IQ projection (a.u.)**. It is not calibrated population. They use the principal axis, or I if fewer than two finite points are available. Analyze counts points after the requested skip.

Bars in Live and Analyze are shot SEM: the projected shot sample standard deviation divided by √(shot count). They describe uncertainty of the point mean, not fit error, shot spread or round-to-round spread. Mean-only files still fit without bars, and fewer than two shots give no bar. R² measures explained variation, normalized residual RMS measures mismatch relative to signal range, and the reported ± is fit uncertainty.

Writeback proposes π/π2 gains in copies of the Run pulse, separate x180/x90 module references, and `ext.pi_product = π gain × source pulse length`. The pulse copies need the captured source cfg; compare their waveform and uncertainty before adopting them. For shared controls see [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md).
