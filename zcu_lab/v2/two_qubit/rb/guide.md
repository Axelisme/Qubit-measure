# Two-qubit paired RB and CZ-interleaved RB

**Readers:** operator, agent. Compare paired decay curves to estimate the CZ error.

## Before run

- Check the [position and tone bindings](../guide.md#綁定位置與讀出-tones). Use Rabi-calibrated **Qubit1 Xπ/2**, **Qubit2 Xπ/2**, **Qubit1 Xπ** and **Qubit2 Xπ** from the qubit slots and **Two-line CZ drive** from the coupler slots. Each X pulse uses its qubit's CZ generator. Keep the calibrated shared carrier, envelope time and start, the signed gains and the relative phase.
- **Local Z corrections** reads `$coupler.ext.cz_phase` and checks the ordered component names. Obtain a matching pair from [Local Z](../local_z/guide.md). Use full **Mux readout** with both tones' calibrated GE circles and GE Post radii; repeat GE and GE Post when changing readout.
- **Random Clifford depths** excludes recovery. Use depth 0 as a baseline and sample across the decay; the initial 0, 1, 2, 3 is not a promise of coverage. **Sequence seeds** sorts and deduplicates IDs, as does the depth list. Add independent seeds to sample more sequences; adding Reps or Rounds only reduces counting noise. Check **Relax delay** against qubit relaxation and ringdown.
- Blank **Reset** leaves the initial ensemble. Use ordinary reset or [mux herald](../guide.md#herald-reset-與獨立-heralded-入口) with the same full **Mux readout**. This entry then counts terminal outcomes only for passed shots; [heralded RB](../../heralded/rb/guide.md) keeps all herald outcomes.
- Recipes `two_qubit_rb` and `heralded_rb`: omission of `depths`, `seeds`, `reps` or `rounds` keeps the form value. Null is passed to the form, not replaced by the initial value. If required blanks remain after overrides, `check_ready` fails before Run. See [shared recipe inputs](../../../recipes/two-qubit-guide.md#共用輸入) for bindings, tab reuse and overrides.

## Sequences and cost

Each shot runs **optional Reset → random Clifford prefix → inverse recovery → terminal mux readout**. Reference Cliffords can already contain CZs; the interleaved arm adds one CZ after every random Clifford. Both arms share the prefix drawn from 11,520 two-qubit Cliffords, and each arm and depth uses its own recovery. Frames start at zero. Each CZ applies its local-Z corrections before the next X rotation; the two qubits finish each Clifford before the next starts.

## Interpreting the result

- **Joint ground return**, Live and first Analyze, pools retained terminal P(g,g) by arm and depth. Other counts in the denominator and does not identify a leakage state. Analyze uses the saved run's **Reps × Rounds**; this is not the retained count.
- **Paired decay fit** gives equal weight to each seed, using only those with return data at every depth for both arms. It needs at least four distinct depths and two complete paired seeds. Dots give the means and curves fit `A p^depth + B`.
- Capped bars on pooled views are shot SEM: the 0/1 success sample SD divided by √N retained shots. Fewer than two shots gives no SEM bar. Pale pooled bars and fit point bars give across-seed sample SD, not shot SEM, fit SE or round variation. The fit view uses complete seed pairs; the pooled view uses available seeds at each point.
- For each arm, 1/e depth and its ± fit SE are in Cliffords. Relative error is their ratio, SE/depth. R² measures explained variation; normalized RMS divides residual RMS by measured span. These metrics have no numeric acceptance threshold. Check both curves and the named arm's depth-window warning before using a fit.

CZ error is `3/4 × (1 − p_interleaved/p_reference)`; reference error per Clifford is `3/4 × (1 − p_reference)`. The 95% bootstrap CI resamples whole paired seeds, not individual shots (default 1000 draws, bootstrap seed 42). It omits systematic errors such as readout bias and gate-dependent noise. A negative value diagnoses slower interleaved decay; keep its sign but do not report a physical negative error.

If the first nonzero depth misses the early decay, add shallower depths or check depth 0/1 and the pulse/readout setup. If a resolved 1/e depth of 20 Cliffords lacks tail coverage, extend toward 60 and compare the uncertainty. A broad CI with small shot SEM calls for more independent seeds. The ratio does not specify which CZ knob to change. There is no writeback.

See the [legacy load note](../guide.md#opendefaults-與人工-apply) and [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
