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

## Counts and plots

The joint fit shares phase slope per gate for each actual spectator and frequency, with separate contrast and phase offset per herald. The ee group negates both σ values and swaps the nominal I/Xπ labels to actual e/g. In both groups, `σ=(N_e−N_g)/N_total` before correction and other remains in the denominator.

- **Run and first Analyze quadratures** connect measured gg quadratures for the current or last started train; empty groups stay blank. Further Analyze panels show gg and sign-corrected ee X/Y, contrast and wrapped phase for every planned train. X/Y bars are ±1 shot SEM (sample SD / √N from that herald group's retained shots), not slope uncertainty or round spread. Low or zero acceptance is not incomplete acquisition: both projections need complete Reps × Rounds counted across all 81 classes before selection.
- **Slopes** compares the joint and gg-only fit, with fit SE and reduced χ² as in the [ordinary guide](guide.md#read-phase-and-quadratures). Each individual herald group must retain at least two usable counts with difference gcd 1; combining two aliased sets does not fix either. Missing fits are not zero.
- **Herald heatmap:** panels split nominal spectator and X/Y; MHz runs across each panel. Each vertical gate-count band lists nine herald classes, Qubit1 slowest, Qubit2 fastest. Gray means no probability. The gg fraction measures acceptance, not ground-state purity. Compare it with the quadrature contrast before trusting the slope.
- **π crossing** compares joint and gg-only conditional phase (spectator-e slope minus spectator-g). Only a unique unambiguous joint crossing supports a proposal; no gg fallback or interpolation uncertainty. Missing, flat, ambiguous or multiple crossings block it.

## Next step

Writeback substitutes the joint crossing frequency in both CZ pulses and offers their two separate coupler slot rows. Recheck amplitude and local Z after accepting. Heralding does not prove gate fidelity. Older files and missing source cfg: see the [family load note](../guide.md#opendefaults-與人工-apply). For common GUI actions, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

In the wrapped phase, per-CZ slope and π-crossing panels, a break between adjacent samples can mark the ±180° cut or an ambiguous half-turn: angles separated by 180° or more are not joined. A frequency with no available value has no point or bridging line. The red π marker is the crossing from joint interpolation, not the gg-only trace.
