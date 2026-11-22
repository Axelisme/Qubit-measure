# Constructive relative phase

**Readers:** operator, agent. Find the relative phase that maximizes the return oscillation rate for a fixed two-line drive pair.

## Sequence

Each shot runs **optional Reset → Xπ on both qubits → simultaneous two-line drive → double Xπ contrast → terminal mux readout**. Scan Qubit2 phase relative to Qubit1 and the common envelope duration. The return is a classified proxy, not a gate fidelity or direct leakage population.

## Prepare the form

- Check [bindings and tone order](../guide.md#綁定位置與讀出-tones), calibrated **Qubit1 Xπ**, **Qubit2 Xπ**, and full **Mux readout** with matching GE circles and GE Post radii.
- **Dual-line drive** starts from the coupler's `cz_qubit1` and `cz_qubit2` slots. Without them, fill the cosine templates (RF frequency, signed gain, NQZ and envelope duration) on the captured qubit channels. Use independent generators with matching frequency, length and pre-delay; the sequence waits for both post-delays. Choose the carrier from conditional spectroscopy and keep each signed gain fixed.
- **Qubit2 relative to Qubit1 drive phase** (deg) sets `Qubit2.phase = source Qubit1.phase + sweep phase`; Qubit1 remains the reference. Sample enough distinct phases to resolve a maximum. **Common waveform envelope duration** (µs) must be positive and cover an oscillation; keep the spacing < `1/(2 × fastest expected rate)`. Without resolved rates there is no phase candidate.
- Blank **Reset** omits reset. To filter shots, use [mux herald and check its wait](../guide.md#herald-reset-與獨立-heralded-入口). Coverage can precede completion of all rounds; check retained shots.
- Recipe `two_qubit_constructive_phase`: omission of scan or acquisition inputs keeps the form values. Null `relative_phase_range` or `duration_range` clears its Start and Stop; null `relative_phase_points`, `duration_points`, `reps` or `rounds` passes to the form. Unrepaired required blanks fail `check_ready` after overrides, before Run. See [shared recipe inputs](../../../recipes/two-qubit-guide.md#共用輸入) for bindings, tab reuse and overrides.

## Read the figures

- **Run and Analyze maps:** terminal P(g,g) over relative phase and duration, out of all nine retained classes. Run shows progress. The **terminal probabilities** trace uses the measured phase nearest the proposal (first measured phase if none) and combines all outcomes except g,g and e,e into one trace. It does not identify any one higher-level state.
- **Duration slices** compare measured return and cosine fits. Each trace needs data with retained shots at all durations. Measured probabilities here and in the terminal trace use ±1 shot SEM (sample SD of the class indicator / √N retained shots), not fit SE or round variation. Maps have no cell bars.
- **Duration-fit rates**: a rate below 1/duration span is marked unresolved and excluded from the interference fit. Fit SE bars on selected rates are in MHz; a missing fit is not rate zero. **Rate² versus relative phase** uses the propagated bars `2 × rate × rate SE` in MHz² for display, not for weighting.

## Review and apply phase proposals

The fit `rate² = A + B cosφ + C sinφ` gives the maximum-rate phase `atan2(C,B)`. It needs more than three identifiable phase slices with full-rank coverage and finite uncertainty; zero modulation also blocks a proposal. R² measures explained rate² variation, normalized RMS divides the residual RMS by the measured rate² span, and relative parameter SE divides SE by fitted magnitude. These do not set an extra cutoff.

Writeback changes only the Qubit2 CZ pulse phase to source Qubit1 phase plus the fitted relative phase and offers the independent `cz_qubit2` coupler slot row. This is not a logical local-Z correction. Use the accepted phase with [Chevron](../chevron/guide.md) to choose the frequency and duration window.

For older files see the [family load note](../guide.md#opendefaults-與人工-apply). For common GUI actions, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
