# Heralded Two-qubit paired RB and CZ-interleaved RB

**Readers:** operator, agent. Run heralded paired two-qubit RB and interpret gg/ee return survival alongside CZ-interleaved decay fits.

**Last updated:** 2026-11-20

Cfg sweep fields use `depth` / `seed`. This experiment uses cfg schema 2.0. Loading a run from an older major version retains Result and metadata, sets cfg to None and emits `CfgUnavailableWarning`. These old runs cannot be reanalyzed. No cfg migration or field alias is provided.

Select `two_qubit/rb_heralded`. Calibrate the X90 pulses, full mux readout,
dual-line CZ pulses and ordered local-Z corrections as described in the
[ordinary RB guide](guide.md). The same component roles, sequence seeds,
depths, bootstrap settings and d=4 error interpretation apply here. Supply
strictly increasing `depth` values; duplicates are listed in the configuration
error before build. Open or Reset captures sources. Refresh preserves authored inputs.

## Shot sequence

Each shot emits optional reset, herald, the paired Clifford sequence with
its own recovery, then terminal readout. Set `reset=None` to omit reset.
Herald and terminal emit the same full mux module with tags `herald` and
`terminal`. Readout `post_delay` is the wait after herald. The two reads
cannot use different gains or integration lengths. The recovered sequence
has net logical identity. A herald gg state should return to gg, and an ee
state should return to ee.

Both F1 and F2 tones require circle classifiers with radius from GE Post.
Missing calibration, missing radius or linear classification fails before Run.
Repeat GE and Post after changing readout settings. Classification happens
during acquisition for both tagged reads, with g, e and other outcomes.

## Sequences and cost

Since 2026-11-20, both arms align the two qubit timelines at every Clifford
boundary, including before recovery, as in ordinary RB. The next Clifford
waits until both timelines finish their pulses and configured delays. Within
a Clifford, each qubit keeps its X90 and virtual Z order and plays X90s as soon
as its own timeline allows. With zero X90 delays,
`X90_Q0, Z90_Q1, X90_Q1, X90_Q0, CZ` starts the first Q0 and Q1 X90s together;
Q0's second X90 follows its first. CZ still waits for both qubits, then starts
both charge-line pulses together. Both qubits resume at the later CZ end,
including delays. Terminal readout waits for both qubits after recovery.

Virtual Z still advances only its own logical frame. X90 phase remains the
configured phase minus that frame. CZ keeps its physical charge phases fixed
and applies the same local-Z frame corrections afterward. Each arm retains
its own full-product recovery. Clifford depth and random seed identities do
not change with the schedule.

The inserted CZs already aligned the interleaved arm, so its pulse times
are unchanged. The reference arm no longer overlaps neighboring Cliffords
and can take longer. Do not directly pool or compare error estimates from
this schedule with those from older schedules, including the concurrent
schedule that allowed overlap across Clifford boundaries. Overlap within a
Clifford can still produce crosstalk in both arms; the comparison does not
show that it cancels.
Herald filtering and optional reset are unchanged. Post-selection is not
physical reset and does not prove a particular initial-state purity.

## Counts and complete rounds

Depth, seed and arm stay on the host because each changes the gate sequence.
Each depth/seed/arm compiles once. The host acquires `rounds` separate rounds,
each using `rounds=1` and `reps` shots. Result `sums` has float64 shape
`(Nd, Nseed, 2, Nround, 3, 3, 3, 3)`. The trailing axes are herald F1,
herald F2, terminal F1 and terminal F2. Each is ordered g, e, other.
Save retains all 81 unnormalized class counts. Storage does not grow with
`reps`. Unfinished rounds remain NaN. Stop retains completed slots;
acquisition errors still need resolution. Native Load preserves the saved
source without changing the current Form. Old raw-IQ files lack `sums`
and are rejected.

Analysis uses two herald slices. gg survival is terminal gg divided by
all terminal counts accepted as herald gg. ee survival is terminal ee divided
by all terminal counts accepted as herald ee. Terminal other remains in
both denominators. Zero accepted counts give NaN survival, even in an acquired
round. `acquired_rounds` records whole acquired 81-class slots independently
of this survival availability.

Each herald selects its own complete seed/round pairs across both arms and
every requested depth. The two selections need not intersect.
Complete rounds average within each seed, then seeds receive equal weight.
Counts determine binomial point errors, not the relative weights of seeds.
Errors propagate through both averages. All-success or all-failure points
use a finite-shot variance floor of `1/(4*N**2)` before averaging.

## Joint decay and uncertainty

Reference and interleaved each have one fitted decay shared by gg and ee.
Each herald and arm has its own amplitude A and offset B in
`A*p**depth+B`. The two fit figures show each herald's equal-seed means,
propagated binomial error bars and joint decay curves. The live plot instead
shows count-weighted gg and ee return survival in both arms.

The fit requires at least four distinct depths and at least two valid paired
seeds per herald. The 95% interval resamples seed identities from the union
of valid gg and ee seeds. Each herald uses only its valid identities from
that same draw. Repeated draws retain their multiplicity. A draw with fewer
than two retained seed entries in either group fails. At least 90% of the
requested bootstrap fits must succeed, with at least 100 requested draws.
The default is 1,000 draws with bootstrap seed 42. This interval describes
statistical seed variation, not readout errors or model systematics.

The d=4 estimates are `reference_epc=3/4*(1-p_reference)` and
`gate_error=3/4*(1-p_interleaved/p_reference)`. Negative error estimates
remain negative. Analyze returns only `estimate`, `fit_quality` and `reason`.
This analysis does not report ordinary RB's joint00 tail warning.
Fit quality retains residual diagnostics for gg reference/interleaved and ee
reference/interleaved. The estimate's paired seed and round counts refer to
the union. Per-herald pair masks, survival, curves and herald fractions stay
inside analysis for plotting. Missing saved cfg,
insufficient paired data, flat decay, invalid fits or unstable bootstrap
leave an explicit reason and no estimate. A finite error remains a
model-dependent diagnostic, not physical calibration acceptance.

## Herald fractions

The herald figures show all nine herald proportions for each arm and depth,
aggregated over seeds and rounds. All herald ge, eg and other outcomes remain
saved, but do not enter the joint fit. The analysis also retains per-round
herald proportions and pass fraction. Pass fraction is herald gg counts
divided by all herald counts. It measures classified acceptance, not
ground-state purity. Readout errors can accept excited states and reject
ground states. Post-selection reduces the available shot count.

Compile-only checks do not validate numerical fitting, save/restore or plot
appearance. Those require the separate offline validation.

## Notebook

Use a validated `RBHeraldedCfg` as `cfg`. Supply the notebook's existing
hardware handles, device manager and destination `point_view`.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.rb.core_heralded import RBHeraldedExp, RBOptions

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(RBHeraldedExp())
record = exp.run(cfg)
analysis = exp.analyze(RBOptions())
saved = exp.save_run(point_view, source=record)
```
