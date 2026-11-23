# Heralded Conditional phase

**Readers:** operator, agent. Find a π crossing from a joint fit of gg- and ee-heralded Ramsey measurements.

## Inputs and sequence

Select `two_qubit/conditional_phase_heralded` and check these inputs before Run:

- Bind `qubit1` and `qubit2` to distinct qubits in `mux`, and bind `coupler` for the CZ pair. Use the full **Mux readout**. Check the tone for each role against the mux order. Both role tones need GE circle classifiers and GE Post radii calibrated with these readout settings.
- Supply calibrated Xπ and Xπ/2 pulses for each role and both CZ drives. **Dual-line drive** starts from the coupler's `cz_qubit1` and `cz_qubit2` slots; fill any missing pulse. Each role's pulses share one generator, distinct from the other role's. Match the CZ envelope lengths and pre-delays. See the [ordinary guide](guide.md#prepare-the-inputs) for context.

Each shot runs **optional ordinary Reset → herald mux readout → spectator I/Xπ → target Xπ/2 → n CZ pairs → independent X or Y projection → terminal mux readout**. Both reads use the same full mux settings. Blank Reset omits the optional pulse but not herald; do not select a mux-herald reset module.

Set **Ramsey target**, **CZ gates per shot** (at least two unique positive counts with difference gcd 1, e.g. [1,2,3]) and increasing **Common CZ drive frequency** (MHz) across the expected crossing. [Recipe blank inputs](guide.md#prepare-the-inputs) follow the same rules as ordinary phase.

## Wait after herald

Set **Mux readout** post-delay from `5/(2π·κ)` µs with the smallest FWHM linewidth κ of the shared readout, in MHz. The wait starts at the end of the generator waveform, not the ADC window, and both reads share it. Residual photons can detune the next qubit pulse. In [Joint readout](../joint_readout/guide.md), use gg herald and sum the target terminal-e columns in the row with target Xπ and spectator I. Lengthen only post-delay and repeat; if the e fraction still rises, adjust the wait before trying to resolve a π crossing. This comparison is not a fidelity estimate.

## Read the joint conditional phase

The joint fit shares a phase slope per gate for each actual spectator and frequency. It fits a separate contrast and phase offset for each herald group. The ee group negates both σ values and swaps the nominal I/Xπ labels to actual e/g. Both groups use `σ=(N_e−N_g)/N_total` before this correction; other stays in the denominator.

**Conditional phase** shows the three panels described in the [ordinary guide](guide.md#read-the-conditional-phase). The Bloch vectors and accumulated-phase markers show gg measurements. The fitted lines use the joint slopes and the fitted gg offsets. The frequency scan subtracts the joint spectator-g slope from the joint spectator-e slope. It alone determines the π crossing and proposal status; the gg-only fit is not a fallback.

**Live** shows the same joint phase per CZ and 180° reference. A frequency needs complete Reps × Rounds in all planned projections, counted across all 81 classes, before it can contribute. Zero accepted shots in a herald group is not incomplete acquisition, but can leave its slope unavailable.

## Inspect diagnostics

- **Herald gg / Herald ee, sign corrected** show X/Y by actual spectator for each train. **Phase points** shows each group's contrast and wrapped single-train angle. X/Y bars show ±1 retained-shot SEM; angle bars propagate each group's X/Y count variance. A calculable angle can have no bar if either projection retains fewer than two shots. Missing X/Y or zero contrast leaves no angle.
- **Slope diagnostics** compares joint and gg-only spectator slopes, their fit SE and reduced χ². The bottom panel compares both conditional-phase curves in the 0–360° view. Each individual group needs usable gate counts with difference gcd 1; combining two aliased sets does not fix either.
- **Herald** splits the initial readout distribution by nominal spectator and X/Y. Each gate-count band lists nine herald classes in Qubit1-major order. All panels share the measured range. Gray means no probability is available. The gg fraction measures acceptance, not ground-state purity. Compare it with the quadrature contrast before trusting the slope.

## Next step

Writeback substitutes the joint crossing frequency in both CZ pulses and offers their two separate coupler slot rows. Recheck amplitude and local Z after accepting. Heralding does not prove gate fidelity. Older files and missing source cfg: see the [family load note](../guide.md#opendefaults-與人工-apply). For common GUI actions, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
