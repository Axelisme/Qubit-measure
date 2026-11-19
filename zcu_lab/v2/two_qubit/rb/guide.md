# Two-qubit paired RB and CZ-interleaved RB

**Readers:** operator, agent. Run paired two-qubit RB and CZ-interleaved RB through the GUI or MCP and interpret their error estimates.

Cfg sweep fields use `depth` / `seed`. This experiment uses cfg schema 5.0. Loading a run from an older major version retains Result and metadata, sets cfg to None and emits `CfgUnavailableWarning`. These old runs cannot be reanalyzed. No cfg migration or field alias is provided.

**Last updated:** 2026-11-20

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Before run

Both Forms list reset, X90, dual-line drive and mux readout first.
Captured roles and the phase pair follow. The host scan inputs appear as outer
`depth`, then inner `seed`. Acquisition settings come last.

Select `two_qubit/rb` after calibrating the F1/F2 X90 pulses, complete mux readout,
two charge-line CZ pulses and their local-Z corrections. Bind scalar aliases
`f1`, `f2`, `coupler` and `mux`, then Open or Reset the Form. The fixed capture
maps logical Q0 to F1 and Q1 to F2. Their tone positions come from the captured
MuxGroup `qubits` order. Use those positions to identify the roles.

Readout references `$mux.module.readout`, preserving the library's complete tones and scalar settings. A missing readout slot stays Invalid.
The Form copies `$f1.module.x90`, `$f2.module.x90`,
`$coupler.module.cz_f1` and `$coupler.module.cz_f2`. Missing pulse slots stay Invalid.
Supply measured calibration for every required slot. The ordinary
CZ pulses use independent generators, the same frequency and envelope duration,
and a common start. Their individual gains and constructive relative phase
remain unchanged throughout a sequence.

The required `$coupler.ext.cz_phase` input contains `f1`, `f2`,
`correction_f1_deg` and `correction_f2_deg`. Names must match the ordered source
F1/F2 components. Swapping the roles invalidates the pair. Corrections are
logical Z angles in degrees, normally minus the corresponding measured local-Z
slope. Missing angles remain invalid. Review any operator override against measured
calibration before Run; model hypotheses only describe assumed behavior.

The saved readout cfg must supply circle g/e centers and radius at
`roles.f1_tone` and `roles.f2_tone`, not by ADC sorting. Run rejects missing,
linear or radius-free classifiers and requests GE Post. After changing readout
settings, repeat GE and Post before using the classifiers.

Reset defaults to `None`. To emit a preparation pulse before each sequence,
select an ordinary reset. To keep only shots classified as g on both role
tones before the sequence, select a `reset/mux_herald` module in Reset.
The Form does not offer `reset/herald`, which requires a pulse readout.
The module's `readout` must match the terminal Readout field, including the
classifiers and the full tone order. Reference the same library key in both
fields. For F1 and F2 at positions 0 and 1 in the full tone list, the module is:

```yaml
type: reset/mux_herald
readout: MG12.readout.main
keep: {'0': g, '1': g}
```

Use the actual F1 and F2 tone positions, not ADC numbers. Quote both keys.
An optional `prepare` emits one ordinary reset or pulse before the herald read.
The mux readout's delays apply to both reads. RB saves only terminal counts
from shots that pass both herald classifications. To preserve all herald
outcomes and fit gg and ee return survival jointly, use the separate
[`two_qubit/rb_heralded`](guide_heralded.md) experiment.

## Sequences and cost

`depth` is an increasing list of distinct nonnegative Clifford counts.
`seed` is a list of distinct nonnegative 31-bit integers. Each seed fixes the
random Clifford draws. Keep these values with the data. Reusing a seed at
shallower depths keeps the same random Clifford prefix.

Both arms draw from all 11,520 two-qubit Cliffords. The decomposition minimizes
CZ count first, then X90 count. Z90 only updates the frame and has no cost.
The uniform average is 1.5 CZs per Clifford, before interleaving and recovery.
The same seed still draws the same Cliffords as with the previous decomposition,
but their pulse sequences can differ.

Interleaved inserts one CZ after each random Clifford. Reference already
contains CZs inside the Clifford decompositions. Each arm gets its own
full-product inverse, including all inserted CZs. Depth counts Cliffords;
physical CZ and pulse counts depend on the selected decompositions and arm.
Check representative compiled sequences for memory use and duration before a run.
A shallow depth can still contain several long CZ pulses.

Every depth/seed/arm starts with zero logical frames. Positive logical Z90
changes a later X90's programmed phase by minus 90 degrees. CZ playback
keeps its two charge phases fixed while logical frames evolve separately.
After CZ, the two named logical corrections advance the respective frames.
Virtual Z updates the logical frame only. Computational-basis readout is
invariant to terminal Z frames.

Since 2026-11-20, both arms in this experiment and the heralded variant align
the two qubit timelines at every Clifford boundary, including before recovery.
The next Clifford starts only after both timelines finish their pulses and
configured delays. Within a Clifford, each qubit keeps its X90 and virtual Z
order and plays X90s as soon as its own timeline allows. With zero X90 delays,
`X90_Q0, Z90_Q1, X90_Q1, X90_Q0, CZ` starts the first Q0 and Q1 X90s together.
Q0's second X90 follows its first.
CZ still waits for both qubits and starts both charge-line pulses together.
Both qubits resume at the later CZ end, including delays. Terminal readout
waits for both qubits after recovery.

The inserted CZs already aligned the interleaved arm at these boundaries, so
its pulse times are unchanged. The reference arm no longer overlaps
neighboring Cliffords and can take longer. Do not directly pool or compare
error estimates from this schedule with those from older schedules, including
the concurrent schedule that allowed overlap across Clifford boundaries.
Overlap within a Clifford can still produce crosstalk in both arms; the
comparison does not show that it cancels.

Each depth/seed/arm has one compiled program and `rounds` host acquisitions of
`reps` shots. Every host acquisition uses `rounds=1` to preserve its own slot.
Without reset, the nominal B00 start leaves the thermal initial population in
place. Terminal F1/F2 classification happens during acquisition. Circle outcomes
are g, e and other. Radius boundaries and equidistant points are other.
The live plot shows joint00 from accepted counts accumulated over seeds and
rounds in each arm. This count-weighted view is not the equal-seed decay fit.

## Save, cancel and analyze

Result `sums` is float64 with axes `depth, seed, arm, round, F1 class, F2 class`
and shape `(Nd, Nseed, 2, Nround, 3, 3)`. Arm 0 is reference and arm 1 is
CZ-interleaved. Each class axis is ordered g, e, other. Counts preserve joint
outcomes from the same accepted shot. They are not marginal products or probabilities.
Heralding adds no saved axis or field, and storage does not grow with `reps`.
At each depth, seed and arm with all rounds finished, sum the class counts over
rounds and divide by `reps × rounds` to obtain the herald acceptance fraction.

Cooperative Stop preserves completed whole rounds in fixed slots. Unfinished
rounds stay NaN. `stop_requested` records a stop request, not data completeness.
Acquisition exceptions propagate unchanged from Run. A failed Run publishes no
new record and leaves the previous successful record unchanged, even after Stop.
Save counts and cfg before closing the source. Native Load restores that
source for Analyze and leaves the current Form unchanged. Old per-shot IQ
files are rejected because they lack `sums`; there is no conversion fallback.

Analyze divides each round's terminal gg count by the sum of accepted counts,
including other. It does not classify IQ again. A point is complete once every
round has run, even if some rounds accepted no shots. A zero-acceptance round
has undefined survival, so the fit excludes that seed/round pair if either arm
has zero acceptance at any depth. Malformed or partially populated count rounds
cause an error.

The decay fit uses only seed/round pairs with finite survival across both arms
and every requested depth. Analyze averages those rounds within each seed and
then weights seeds equally. It requires at least two paired seeds and four
distinct depths. An incomplete run can meet those requirements if it retains
enough complete curves.

## Interpreting the result

Analyze returns only `estimate`, `fit_quality` and `reason`. Per-depth survival,
round masks and fitted curves stay inside analysis for plotting.

Paired decay analysis fits each arm to `A*p**depth+B`.
The two-qubit dimension is 4. The reported quantities are
`reference_epc = 3/4*(1-p_reference)` and
`gate_error = 3/4*(1-p_interleaved/p_reference)`.
Negative error estimates remain negative. They are diagnostics of the ratio
and its assumptions. Interpret a negative estimate as a model diagnostic,
while retaining the reported value.

The 95% interval resamples entire paired seed curves, preserving depth and arm
correlations. It describes statistical seed variation; independent-shot
uncertainty requires a different model. The summary reports paired seed/round
counts and valid bootstrap fits. The default is 1,000 deterministic resamples with bootstrap seed 42.

The fit plot shows equal-seed paired means and available fitted decays.
The summary includes each arm's R² and normalized residual RMS. Parameter
covariance and marginal parameter errors are unavailable; use the paired-seed interval and residual diagnostics for manual model review.
Insufficient data, absent saved cfg, flat decay, invalid fits or unstable bootstrap
stop estimation and retain the failure reason. Preserve that result and review
the data before deciding to remeasure. Statistical diagnostics and physical
acceptance remain operator decisions.

Thermal population, preparation and readout errors contribute to this ensemble.
Circle other outcomes do not identify a particular leakage state. Finite shots
and too few shallow depths can leave the decay unidentifiable. A shallow
curve can also yield finite but biased parameters when amplitude and offset
are poorly constrained. Even a finite fit needs an identifiability review.
Gate-dependent noise, coherent errors and nonexponential decay can bias the
interleaved ratio beyond its statistical interval. Inspect the curves and those limitations before
interpreting a CZ error. This experiment provides diagnostic estimates only;
physical calibration acceptance is manual.

## Notebook

Use a validated `RBCfg` as `cfg`. Supply the notebook's existing hardware
handles, device manager and destination `point_view`.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.rb.core import RBExp, RBOptions

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(RBExp())
record = exp.run(cfg)
analysis = exp.analyze(RBOptions())
saved = exp.save_run(point_view, source=record)
```

The adapter owns the run record and plots for each operation.

## Calibration order and field units

Calibrate each role's Xπ/2 slot with amplitude and length Rabi before using it.
Confirm pulse frequency in MHz, signed gain in [-1, 1], and duration in us.
Use lookback to set mux timing, then mux onetone to set frequencies and gains.
Confirm readout timing in us and gains in [-1, 1].
Locate the dual-line CZ pulse with conditional spectroscopy and chevron.
Confirm the coupler's `cz_f1` and `cz_f2` slots before copying them into the Form.
