# Conditional phase

**Readers:** operator, agent. Find a common CZ carrier whose conditional phase per gate crosses π.

## Prepare the inputs

- Check [position bindings and mux tone order](../guide.md#綁定位置與讀出-tones). Use calibrated Xπ and Xπ/2 slots for each qubit and full **Mux readout** with both role tones' GE circles and GE Post radii.
- **Dual-line drive** copies the coupler's `cz_qubit1` and `cz_qubit2` pulses; supply missing slots or fill the required pulses. Each role's Xπ, Xπ/2 and CZ use the same generator, distinct across roles. Both CZ lines need matching envelope length and pre-delay; the gate waits for both. The common sweep supplies both carriers. Keep the signed gains, relative phase and waveform fixed.
- **Ramsey target** receives Xπ/2; the other qubit is spectator. Set **CZ gates per shot** to at least two unique positive integers with difference gcd 1 (default [1,2,3]); [1,3,5] aliases slopes a half-turn apart and is rejected. Use an increasing **Common CZ drive frequency** sweep in MHz around the expected crossing. Spacing must resolve phase changes smaller than a half-turn between points. Both carriers are replaced at each frequency.
- Blank **Reset** skips reset. This entry accepts ordinary reset but not mux herald. For joint gg/ee conditioning choose [Heralded Conditional phase](guide_heralded.md).
- Recipes `two_qubit_conditional_phase` and `two_qubit_conditional_phase_heralded`: omission of measurement inputs keeps the form value. Null `frequency_range` clears both frequency endpoints; null `frequency_points`, `gate_count`, `ramsey_role`, `reps` or `rounds` passes to the form. Required blanks left after overrides fail `check_ready` before Run. See [shared recipe inputs](../../../recipes/two-qubit-guide.md#共用輸入) for bindings, tab reuse and overrides.

## Sequence and counts

Each shot runs **optional Reset → spectator I/Xπ → target Xπ/2 → n two-line CZ gates → X or Y projection → terminal mux readout**. The independent projections use the preparation phase φ or φ+90°. The two spectator labels describe nominal preparations, not verified pure states.

## Read phase and quadratures

- **X/Y panels** connect measured `σ=(N_e−N_g)/N_total` versus carrier by spectator and train length. Other contributes zero and stays in the denominator. Bars are the retained-shot sample SD of this indicator divided by √N, not fit SE or round spread. Run and the first Analyze panel show the current or latest started train, even if incomplete. Planned trains also have separate panels.
- **Contrast and wrapped phase:** `hypot(σ_X,σ_Y)` and `atan2(σ_Y,σ_X)` in degrees; no angle for zero contrast. Each projection must have all Reps × Rounds to enter the phase calculation. Counts from completed rounds remain visible after Stop; incomplete frequencies can have slope diagnostics but block crossing selection.
- **Slopes:** X and Y fit `A cos(n·slope+φ)`, `A sin(n·slope+φ)` with no DC term, separately by spectator and frequency. Effective fit groups still need difference gcd 1 after dropping unusable points. Bars are slope fit SE; reduced χ² is the sum of squared normalized residuals divided by residual degrees of freedom. Near 1 is consistent with the fit errors, not a selection threshold. A missing fit or χ² is not zero.

In the wrapped phase, per-CZ slope and π-crossing panels, a break between adjacent samples can mark the ±180° cut or an ambiguous half-turn: angles separated by 180° or more are not joined. A frequency with no available value has no point or bridging line.

## Select a frequency and apply

**π crossing** plots the difference of spectator slopes, e minus g, not the phase of a chosen train. Analyze unwraps adjacent frequencies and linearly interpolates one π-modulo-360° crossing. Missing slopes, no root, multiple roots, a flat π segment or a half-turn ambiguity block the proposal. The red marker shows the interpolated crossing. A boundary root counts once; interpolation gives no frequency uncertainty.

Writeback replaces only the common frequency of both source CZ pulses with that crossing and offers independent coupler slot rows. Recheck amplitude after accepting a new carrier and then measure [Local Z](../local_z/guide.md). Phase analysis requires the saved cfg; see the [legacy load note](../guide.md#opendefaults-與人工-apply). For common GUI actions, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
