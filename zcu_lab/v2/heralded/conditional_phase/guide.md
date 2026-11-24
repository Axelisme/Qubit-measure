# Heralded conditional phase

**Readers:** operator, agent. Find a π crossing from a joint fit of gg, ge, eg and ee Ramsey measurements.

## Inputs and sequence

Select `heralded/conditional_phase` and check these inputs before Run:

- Bind `qubit1` and `qubit2` to distinct qubits in `mux`, and bind `coupler` for the CZ pair. Use the full **Mux readout**. Check the tone for each role against the mux order. Both role tones need GE circle classifiers and GE Post radii calibrated with these readout settings.
- Supply calibrated Xπ and Xπ/2 pulses for each role and both CZ drives. **Dual-line drive** references the coupler's `cz_qubit1` and `cz_qubit2` slots; fill any missing pulse. Each role's pulses share one generator, distinct from the other role's. Match the CZ envelope lengths and pre-delays. See the [ordinary guide](../../two_qubit/conditional_phase/guide.md#prepare-the-inputs) for context.

Each shot runs **optional ordinary Reset → herald mux readout → spectator I/Xπ → target Xπ/2 → n CZ pairs → independent X or Y projection → terminal mux readout**. Both reads use the same full mux settings. Blank Reset omits the optional pulse but not herald; do not select a mux-herald reset module.

Set **Ramsey target**, **CZ gates per shot** (at least two unique positive counts with difference gcd 1, e.g. [1,2,3]) and increasing **Common CZ drive frequency** (MHz) across the expected crossing. This sweep sets both carriers and ignores the frequencies saved in the library. [Recipe blank inputs](../../two_qubit/conditional_phase/guide.md#prepare-the-inputs) follow the same rules as ordinary phase.

## Wait after herald

Set **Mux readout** post-delay from `5/(2π·κ)` µs with the smallest FWHM linewidth κ of the shared readout, in MHz. The wait starts at the end of the generator waveform, not the ADC window, and both reads share it. Residual photons can detune the next qubit pulse. In [Joint readout](../../two_qubit/joint_readout/guide.md), use gg herald and sum the target terminal-e columns in the row with target Xπ and spectator I. Lengthen only post-delay and repeat; if the e fraction still rises, adjust the wait before trying to resolve a π crossing. This comparison is not a fidelity estimate.

## Read the joint conditional phase

For each actual spectator state and frequency, the four herald groups share a phase slope per CZ gate. Each group has its own contrast and phase offset. Start with `σ=(N_e−N_g)/N_total` for each X or Y projection. If the **target** heralded e, negate both projections. If the **spectator** heralded e, put nominal I in the actual e branch and nominal Xπ in the actual g branch. These two corrections depend on the selected Ramsey target. Terminal Other stays in the denominator; a shot with either qubit heralded Other does not enter the fit.

**Conditional phase** shows three panels. The first two show pooled Bloch vectors and accumulated-phase points at the marked frequency. For each actual spectator, sum the aligned counts from completed X or Y projections before dividing by their retained shot totals. The plotted prediction combines the four fitted X/Y vectors using the measured shot fractions separately for X and Y, then takes the angle. Between measured gate counts it interpolates the fractions linearly. The points use the 360° branch nearest that prediction. Analyze does not refit the pooled points. Without a fit, the phase trace connects adjacent valid points only when they do not cross the branch cut or make a half-turn jump.

The third panel subtracts the spectator-g slope from the spectator-e slope. If any herald group cannot support a slope at a spectator and frequency, that slope is absent. Both slopes are needed to show the conditional phase at a frequency. A missing phase anywhere in the frequency window rules out a π proposal, even if two other measured points bracket π. The gg-only curve is a diagnostic, not a fallback.

**Phase points** shows pooled X/Y contrast and wrapped phase versus frequency for each CZ gate count. Compare the spectator traces to see whether an apparent phase change occurs where contrast drops or points are missing. **X/Y contrast (Live)** shows the pooled contrast for the latest train with a completed X/Y pair. Its title gives the number of CZ gates per shot in that train. Both show one trace for each actual spectator state.

**Live** also shows the same joint phase per CZ and 180° reference as the third panel. The host runs one CZ gate count at a time. Each completed hardware round supplies the entire frequency axis for that CZ train. When the last train has all planned spectator and X/Y projections at full Reps × Rounds, counted across all 81 classes, the frequencies can appear as a batch. A projection can be fully acquired with zero retained shots in one group. That group still needs usable X/Y data at enough gate counts for its slope fit. The current Live curve does not support widening the frequency window or deciding to stop early during the sweep.

## Inspect diagnostics

- **Aligned target projections by herald** shows measured X/Y by actual spectator for each train, including acquired projections from incomplete trains. **Phase diagnostics** compares each group's contrast and wrapped single-train angle in separate spectator panels. **Accumulated phase diagnostics** shows one panel per herald at the marked sample frequency. Compare each group's phase points with its line, which uses that group's offset and the shared slope. Where a fit is absent, inspect the measured points instead.
- X/Y bars show ±1 retained-shot SEM. The angle and contrast bars propagate shot variance from the independent X and Y projections, using pooled counts on pooled plots and separate counts on group plots. At zero contrast the contrast bar uses the larger projection SEM, and the angle is undefined. Missing X or Y leaves no contrast or angle. If either projection has fewer than two retained shots or its variance is unavailable, the figure omits the bar but keeps any calculable point. These bars do not include classifier or fit uncertainty.
- **Slope diagnostics** compares joint and gg-only spectator slopes, their fit SE and [reduced χ²](../../two_qubit/conditional_phase/guide.md#read-phase-and-quadratures). The bottom panel compares both conditional-phase curves in the 0–360° view. Each individual group needs usable gate counts with difference gcd 1; combining two aliased sets does not fix either.
- **Herald** splits the initial readout distribution by nominal spectator and X/Y. Each gate-count band lists nine herald classes in Qubit1-major order. All panels share the measured range. Gray means no probability is available. A herald fraction measures how often that class occurs, not state purity. Compare low fractions with the group projections to locate missing or weak contrast.

## Next step

Writeback substitutes the joint crossing frequency in both CZ pulses and offers their two separate coupler slot rows. Recheck amplitude and local Z after accepting. Heralding does not prove gate fidelity. Older files and missing source cfg: see the [family load note](../../two_qubit/guide.md#opendefaults-與人工-apply). For common GUI actions, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
