# Heralded Two-qubit paired RB and CZ-interleaved RB

**Readers:** operator, agent. Run heralded paired two-qubit RB and interpret gg/ee return survival alongside CZ-interleaved decay fits.

**Last updated:** 2026-11-20, per-depth seed spread

Cfg schema 3.0 requires `x180_f1` and `x180_f2`.

Select `two_qubit/rb_heralded`. Calibrate X90 and X180 on both roles, full mux
readout, both CZ pulses and ordered local-Z corrections as described in the
[ordinary RB guide](guide.md). Open or Reset captures the same module slots.
The cfg deduplicates and sorts both `depth` and `seed` so that the saved cfg
matches the data axes. Refresh preserves authored inputs.

## Shot sequence

Each shot emits optional reset, herald, the paired Clifford sequence with
its own recovery, then terminal readout. Set `reset=None` to omit reset.
Herald and terminal emit the same full mux module with tags `herald` and
`terminal`. The two reads cannot use different gains or integration lengths.
The recovered sequence has net logical identity. A herald gg state should
return to gg, and an ee state should return to ee.

Both F1 and F2 tones require circle classifiers with radius from GE Post.
Missing calibration, missing radius or linear classification fails before Run.
Repeat GE and Post after changing readout settings. Classification happens
during acquisition for both tagged reads, with g, e and other outcomes.

## Wait after herald

Set the readout's `post_delay` before Run so photons in the resonator can
decay before the qubit drive. Herald and terminal readout share this value
([ADR-0074](../../../../docs/adr/0074-flat-readout-module.md#herald-reset-與逐-shot-遮罩)).
The wait starts when the readout generator waveform ends, without waiting for
the ADC integration window to end
([ADR-0083](../../../../docs/adr/0083-program-macros-and-timelines.md#timeline-只存結尾與-reference-分開)).
No experiment calibrates this field. New mux readout modules default to 0 µs.

If the wait is too short, leftover photons shift the qubit transition
frequency and cause dephasing. A nominal Xπ after herald may barely flip
the qubit. A separate conditional-phase scan may show no π crossing; RB
does not measure that crossing.
In one low-SNR FTF example, after gg herald and a nominal F2 Xπ, the F2
terminal e fraction was 11.6% at 0 µs and 92.9% at 2 µs. These fractions do
not subtract the I baseline. The 2 µs value is an example, not a minimum
or an optimum.

Use each resonator's `kappa` from the `mux/onetone` FIT. It is the full
width at half maximum (FWHM) in MHz. Estimate the cavity field decay time as
`1 / (π · FWHM)` in µs. Start with a wait several times that estimate.

In `two_qubit/joint_readout`, choose `reset/mux_herald` in Reset. Set
`keep` to g for both role tones, keyed by their positions in the full readout.
Use the same full mux readout for herald and terminal. After Run, inspect the
4×9 row-normalized probability matrix in FIT. Find the row with nominal Xπ
on the target and I on the other qubit.
Sum the columns where the target's terminal class is e. Compare with the
gg row for an I baseline if needed. This is the terminal e fraction among
shots that passed gg herald, not gate fidelity. Lengthen only `post_delay`
and repeat Run and FIT. If the longer wait still raises the terminal e
fraction, adjust `post_delay` before running RB.

## Sequences and cost

Each seed draws the deepest random sequence once. Reference and interleaved
play the same Clifford prefixes, with one CZ after each draw in the interleaved
arm. Each depth and arm gets its own recovery. Decomposition minimizes CZs,
then physical X90 and X180 pulses. Virtual Z90 and local-Z corrections advance
each role's frame. Both X pulses play at their configured phase minus that frame.
Both qubits finish each Clifford before the next one starts, as in ordinary RB.
X180 can change pulse timing in either arm. Do not pool these estimates with
runs from older schedules.

Run compiles one program per seed and arm before acquisition. After optional
reset and herald, each shot plays the shared prefix up to its selected depth.
Other depths skip that point's recovery. The selected shot plays its recovery,
waits for its end and jumps to the common terminal readout. The deepest shot
continues to terminal readout without the final jump.

If any program exceeds PMEM, Run stops before taking data. The error gives the
seed, arm, number of depths, largest depth, words used and board limit. Reduce
the largest depth or use fewer points. In a 4,096-word compile example with
synthetic pulses and no reset, the [ordinary RB program](guide.md#sequences-and-cost)
did not fit both arms for 201 points from 0 to 200, but fit both for 21 points
from 0 to 400. Compile with your own settings; this version also emits herald.

## Counts and complete seeds

Run acquires each seed/arm program once. `rounds` is the number of acquisition
rounds per program. QICK visits each depth once per repetition, giving `reps`
shots per depth per round. The acquisition reducer adds the counts from
finished rounds.

Result `sums` has float64 shape `(Nseed, 2, Nd, 3, 3, 3, 3)`. The axes are
seed, arm, depth, herald F1, herald F2, terminal F1 and terminal F2. Arm 0 is
reference and arm 1 is CZ-interleaved. Each class axis is ordered g, e, other.
Save keeps all 81 unnormalized class counts per cell. Neither `reps` nor
`rounds` adds to the storage size.

Stop keeps counts from finished rounds for the current seed and arm. A cell
with none stays NaN. Native Load restores the saved run without changing the
current Form. Load rejects files with a round axis (wrong number of axes) or
older raw-IQ files without `sums`. It does not convert them.

For each cell, gg survival is the terminal gg count divided by all terminal
counts for herald gg. The ee ratio uses terminal ee and all terminal counts
for herald ee. Terminal other remains in both denominators. A cell with
zero accepted shots has NaN survival even if acquisition ran. A cell with a
positive count from fewer rounds is usable.

Each herald group uses only seeds with finite survival in both arms at all
requested depths. The groups need not use the same seeds. The fit weights
seeds equally and uses each cell's actual accepted shot count to calculate
the binomial point errors. All-success or all-failure points use a
finite-shot variance floor of `1/(4*N**2)` before averaging.

## Joint decay and uncertainty

Reference and interleaved each have one fitted decay shared by gg and ee.
Each herald and arm has its own amplitude A and offset B in
`A*p**depth+B`. The two fit figures show each herald's equal-seed means,
seed-spread error bars and joint decay curves. The live plot still shows
count-weighted gg and ee return survival in both arms.

At each depth, the live and fit bars extend one sample standard deviation
(`ddof=1`) across seeds on either side of the point. A point with only one
seed has no bar. The spread includes sequence-to-sequence variation and shot
noise. For survival probability `P` and `N` accepted shots for that seed and
herald, shot noise is about `√(P(1−P)/N)`. It grows as `P` falls from near one
toward one-half, so increasing spread with depth does not by itself identify
coherent error. The fit still uses the propagated binomial standard errors
as weights, not the plotted seed spread.

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
Fit quality reports separate diagnostics for gg reference/interleaved and ee
reference/interleaved. For each arm, both heralds report the same value in
`fit_quality.relative_parameter_errors["decay_time"]` because they share a
fitted 1/e length. A large value can mean the deepest point does not reach
the tail. Increase the maximum depth and check again. In the [synthetic
comparison](guide.md#interpreting-the-result), increasing the maximum depth
from 30 to 200 lowered relative error from 0.30 to 0.035, although R² was
already 0.993 at depth 30.

The paired seed count refers to the union. Per-herald survival, model curves
and herald fractions stay inside analysis for plotting. Missing saved cfg,
insufficient paired data, flat decay, invalid fits or unstable bootstrap
leave an explicit reason and no estimate. A finite error remains a
model-dependent diagnostic, not physical calibration acceptance.

## Herald fractions

The herald figures show all nine herald proportions for each arm and depth,
from counts pooled across seeds. All herald ge, eg and other outcomes remain
saved, but do not enter the joint fit. Pass fraction is herald gg counts
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
