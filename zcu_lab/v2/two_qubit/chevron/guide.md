# Rabi chevron

**Readers:** operator, agent. Use the joint-return chevron to choose CZ frequency and duration windows.

## Sequence and calibration

Each shot runs **optional Reset → Xπ on both qubits → simultaneous two-line drive → double Xπ contrast → terminal mux readout**. The observable is terminal P(g,g) after double Xπ. It includes thermal population, preparation and readout effects; a valley alone does not identify leakage or set a gate.

## Prepare the form

- Use calibrated **Qubit1 Xπ**, **Qubit2 Xπ** and full **Mux readout** with GE circles and GE Post radii on the two positions. Check the [bindings and tone order](../guide.md#綁定位置與讀出-tones).
- **Dual-line drive** starts from the coupler's `cz_qubit1` and `cz_qubit2` slots. Without them, set signed gain, envelope length and NQZ in the cosine template; channels come from the qubits' wiring. Use independent generators and matching envelope length and pre-delay; the sequence waits for both post-delays. The shared frequency sweep supplies both carriers. Keep the signed gain ratio and relative phase fixed; use [constructive phase](../constructive_phase/guide.md) to choose that phase.
- Center **Common CZ drive frequency** (MHz) on a measured spectroscopy feature. **Common waveform envelope duration** (µs) is the full envelope, not the plateau; use strictly positive endpoints and spacing < `1/(2 × fastest expected rate)`. Span at least one expected oscillation to distinguish it from a slope. The scan changes both frequencies and durations together, not phase or gain.
- Blank **Reset** omits reset; for mux herald, use the [same readout and keep keys](../guide.md#herald-reset-與獨立-heralded-入口). Check Reps × Rounds against the retained counts before using partial data.

## Read the figures

- **Run and Analyze chevrons:** joint return against duration (µs) and drive frequency (MHz), with probability as color. Blank cells have no measured probability. Analyze's white cross is the lowest finite measured return; it selects a slice, not a proposal. The nine terminal maps show which outcomes replace g,g. Each colorbar spans its own outcome’s measured range; compare probabilities on the bars, not colors across outcomes. The denominator includes `other`, a classifier outcome, not identified leakage. Heatmaps have no cell error bars.
- **Duration slices:** first and last frequencies plus the cross's frequency (or the middle index if no point is available), without duplicates. Compare measured return with the cosine fit. If a slice has no fit, the line joins the measured points instead. Bars on measured points are ±1 shot SEM: binomial sample SD divided by √N retained shots, not fit uncertainty or round spread.
- **Duration-fit rates:** a cosine gives the rate at each usable frequency. A rate below the inverse duration span is unresolved (crosses); bars for selected rates are fit SE in MHz. A failed fit does not give rate zero. R² is the explained variation; normalized RMS is residual RMS divided by the observed probability range. Neither is a selection threshold. A duration trace needs retained shots at every point; it can reach that coverage before the planned rounds finish.

## Next step

Use a resolved return valley and neighboring traces to choose the test carrier and envelope window for [CZ amplitude](../cz_amplitude/guide.md), then measure conditional phase. Chevron does not write back. After changing waveform or bias, recheck the gate and local Z.

For older files see the [family load note](../guide.md#opendefaults-與人工-apply). For common GUI actions, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
