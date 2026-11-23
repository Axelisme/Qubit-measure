**Readers:** operator, agent. Calibrate pulse lengths and measure the Rabi rate on a mux group.

# Mux length Rabi

Sweep each selected tone's total drive length at fixed gain and read the oscillation in mean IQ. The cosine fit gives the π and π/2 lengths and the Rabi frequency at that gain. The optional envelope describes loss of oscillation contrast.

**Sequence:** common reset → simultaneous drives at one length per tone → mux readout after the longest branch.

## Key settings

Bind the `mux` alias to a `group/mux` with matching qubits and resonators in readout order. Check the full **Readout** module, including frequencies and ADCs for inactive tones. If there is no readout slot, the inline starting values use group wiring and resonator calibrations; verify gain and timing before Run.

**Tones** chooses what to edit. **Participants** takes a nonempty list of unique zero-based positions in run order, e.g. `[2, 0]`. It selects both drives and emitted readout tones. The `participants` alias seeds those positions; if absent, unbound, or empty, the form starts with all tones. Editing the selection preserves tone values. Fill required pulse and sweep fields for any newly selected tone, using distinct drive generators separate from readout.

Under **Tones → Length**, choose positive **Start** and **Stop** in µs to span an inversion. Stop starts at 4 × the x180 waveform length (1 µs if x180 is missing or Arb); **Expts** is shared. For flat_top, total length includes both ramps and must exceed their duration. Use const, flat_top or a supported sampled waveform, not Arb. If any selected waveform is sampled, Live updates point by point and Stop can leave a partial scan. With only const or flat_top it updates a whole scan per completed round.

Use **Fit decay envelope** for a damped cosine and **Fit phase offset** if the measured phase should float; otherwise it is fixed at 0° with signed amplitude. A fit needs at least six finite points.

In `mux_len_rabi`, omitted inputs retain settings. Null `participants`, `sweeps` (clears all bounds) or `n` leaves missing required values at the pre-Run readiness check unless an override fills them. Null `decay` or `fit_phase` is rejected by Analyze, after acquisition. See the [mux recipe inputs](../../../recipes/mux-guide.md#mux_len_rabi).

## Read the plots and use the result

Live `measurement_i` and Analyze `fit_i` use the original tone index `i`, not the participant-list row. Live shows acquired length points. Analyze adds the fit and marks π length if in range. Check that the points support that inversion; projected mean IQ is not population.

Bars in Live and Analyze are shot SEM: the projected shot sample standard deviation divided by √(shot count). They describe uncertainty of the point mean, not fit error, shot spread or round-to-round spread. Mean-only files still fit without bars, and fewer than two shots give no bar. R² measures explained variation, normalized residual RMS measures mismatch relative to signal range, and the reported ± is fit uncertainty.

Writeback proposes `ext.rabi_freq`, π/π2 length pulse copies and their x180/x90 references, plus `ext.pi_product = source gain × π length`. It does not change the amplitude-calibrated pulses. Preserve the ramps when adopting a new total length. For shared controls see [zcu_lab/v2/operator-workflow.md](../../operator-workflow.md).
