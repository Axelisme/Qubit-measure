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

## Read the conditional phase

**Conditional phase** shows three panels in measurement order:

1. **Bloch vectors** shows the target's measured X/Y vector for each gate count `n`, with separate spectator g and e series. Panels 1 and 2 use the measured frequency nearest the selected crossing. When no crossing is selected, they use the sample closest to 180° in fitted phase, or the middle sample if no fit is available.
2. **Accumulated phase** lifts the measured angles onto the fitted line's 360° branch. A slope is the phase added by each CZ. Point bars are ±1 shot SEM. The legend reports the slope; see **Slope diagnostics** for its fit SE.
3. **Conditional phase per CZ** subtracts the spectator-g slope from the spectator-e slope at each frequency. The 0–360° view puts π at 180°, away from the branch cut. The red × marks the interpolated crossing; the dotted vertical line marks the sampled frequency used in panels 1–2.

**Live** shows panel 3 for complete frequencies. The host runs one CZ gate count at a time. Each completed hardware round supplies the entire frequency axis for that CZ train. When the last train has all planned spectator and X/Y projections at full Reps × Rounds, the frequencies can appear as a batch. A complete frequency still needs both fitted slopes to show a phase. The current Live curve does not support widening the frequency window or deciding to stop early during the sweep.

## Read phase and quadratures

- **Quadratures** connects measured `σ=(N_e−N_g)/N_total` versus carrier by spectator and train length. Other stays in the denominator and contributes zero. Bars show the retained-shot sample SD divided by √N. These plots retain counts from completed rounds after Stop even if the frequency cannot enter the slope fit.
- **X/Y contrast (Live)** shows the latest train with a completed X/Y pair, separately for spectator g and e. Contrast needs both X and Y at a frequency.
- **Phase points** shows contrast `hypot(σ_X,σ_Y)` and the single-train angle `atan2(σ_Y,σ_X)` in [−180°, 180°). Zero contrast leaves no angle. Phase bars propagate the X/Y count variance and omit projections with fewer than two retained shots. They do not include classifier uncertainty. Bars at the axis edge are clipped, not wrapped. In Live and Analyze, contrast bars show ±1 shot SEM from first-order propagation of the independent X and Y projection means. At exactly zero contrast, the bar uses the larger projection SEM. If either projection has fewer than two retained shots or its variance is unavailable, a calculable contrast still appears without a bar. Contrast bars do not include classifier or fit uncertainty.
- **Slope diagnostics** shows individual spectator slopes with fit SE and reduced χ² (sum of squared normalized residuals divided by residual degrees of freedom). The model fits `A cos(n·slope+φ)` and `A sin(n·slope+φ)` to X/Y with no DC term. The retained gate-count differences must still have gcd 1. Reduced χ² near 1 is consistent with the fit errors, not a selection threshold.

Phase plots do not join across their display cut, an ambiguous half-turn, or a missing point.

## Select a frequency and apply

Analyze unwraps adjacent frequencies and linearly interpolates a π-modulo-360° crossing. Missing slopes, no root, multiple roots, a flat π segment or a half-turn ambiguity block the proposal. A boundary root counts once. Interpolation gives no frequency uncertainty.

Writeback replaces only the common frequency of both source CZ pulses with that crossing and offers independent coupler slot rows. Recheck amplitude after accepting a new carrier and then measure [Local Z](../local_z/guide.md). Phase analysis requires the saved cfg; see the [legacy load note](../guide.md#opendefaults-與人工-apply). For common GUI actions, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
