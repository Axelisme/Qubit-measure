# Two-qubit paired RB and CZ-interleaved RB

**Readers:** operator, agent. Run paired two-qubit RB and CZ-interleaved RB through the GUI or MCP and interpret their error estimates.

**Last updated:** 2026-11-20, per-depth seed spread

Cfg schema 6.0 requires `x180_f1` and `x180_f2`.

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Before run

Select `two_qubit/rb` after calibrating the F1/F2 X90 and X180 pulses, complete mux readout,
two charge-line CZ pulses and their local-Z corrections. Bind scalar aliases
`f1`, `f2`, `coupler` and `mux`, then open the tab or choose Defaults. The fixed capture
maps logical Q0 to F1 and Q1 to F2. Their tone positions come from the captured
MuxGroup `qubits` order. Use those positions to identify the roles.

Readout references `$mux.module.readout`, preserving the library's complete tones and scalar settings. A missing readout slot stays Invalid.
The Form copies `$f1.module.x90`, `$f1.module.x180`, `$f2.module.x90`,
`$f2.module.x180`, `$coupler.module.cz_f1` and `$coupler.module.cz_f2`.
Missing pulse slots stay Invalid. Each X pulse uses the generator of its
role's CZ line and a finite scalar phase. Supply measured calibration for
every required slot. The ordinary
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

Enter a nonempty list of nonnegative Clifford counts in `depth`, and a
nonempty list of nonnegative 31-bit integers in `seed`. The cfg removes
duplicates and sorts both lists before Run, so the saved cfg matches the data
axes. Each seed draws one random sequence up to the largest depth. Both arms
use the same draw, and shallower depths use its prefixes.

The draw is uniform over all 11,520 two-qubit Cliffords, with replacement.
The decomposition minimizes CZ count first, then the number of physical X
pulses. X90 and X180 each count as one pulse, and Z90 only updates the
frame. The uniform average is 1.5 CZs per Clifford before interleaving and
recovery. A seed still draws the same Clifford identities as before.

Interleaved inserts one CZ after each random Clifford. Reference already
contains CZs inside the Clifford decompositions. At each depth, each arm plays
one recovery Clifford that inverts the full prefix, including its inserted CZs.
Even a shallow depth can contain several CZ pulses.

Each shot starts with zero logical frames. Z90 adds 90 degrees to one frame.
X90 and X180 play at their configured phase minus that frame. CZ keeps the
two charge-line phases fixed and advances each frame by its local-Z correction.
Terminal Z frames do not affect computational-basis readout.

Both arms align the two qubit timelines at each Clifford boundary, including
before recovery. Within a Clifford, each qubit plays its own pulses and virtual
Zs in order. Pulses on different qubits can overlap. At CZ, both qubits wait,
the two charge-line pulses start together, and both qubits resume at the later
CZ end, including delays. X180 can change pulse timing in either arm. Do not
pool decay estimates with older runs that used two X90s or overlapped
neighboring Cliffords.

Run compiles every seed and arm before taking data. Each program contains
one copy of the deepest Clifford sequence. At each requested depth, the
matching board-loop index plays its recovery, waits until it ends, then jumps
to the common terminal readout. Shots for larger depths skip that recovery.
The deepest shot plays its recovery and continues straight to readout.

If a program exceeds PMEM, Run stops without acquiring shots. The error gives
the seed, arm, depth point count, largest depth, words used and board limit.
Reduce the largest depth or use fewer points. On a 4,096-word profile with
synthetic pulses, no reset and seed 7, 201 points from 0 to 200 took
3,728 words for reference and 4,114 for interleaved, so the sweep did not fit.
The same fixture fit 21 points from 0 to 400 at 2,777 and 3,586 words.
These numbers are compile examples, not limits for every pulse or reset setup.

Run acquires each seed/arm program once. `rounds` is the number of acquisition
rounds per program. QICK visits each depth once per repetition, giving `reps`
shots per depth per round. The acquisition reducer adds counts from finished
rounds.

Without reset, the nominal B00 start leaves the thermal initial population in
place. Terminal F1/F2 classification happens during acquisition. Circle outcomes
are g, e and other. Radius boundaries and equidistant points are other.
The live plot shows joint00 from accepted counts accumulated over seeds and
rounds in each arm. This count-weighted view is not the equal-seed decay fit.

## Save, cancel and analyze

Result `sums` is float64 with shape `(Nseed, 2, Nd, 3, 3)` and axes
(seed, arm, depth, F1 class, F2 class). Arm 0 is reference and arm 1 is
CZ-interleaved. Each class axis is ordered g, e, other. Counts preserve
joint outcomes from each kept shot. Neither the round count nor heralding
adds a Result axis. On a full run, divide the total counts in a cell by
`reps × rounds` to estimate the herald acceptance fraction.

Stop keeps the sum of completed rounds for the current seed and arm, even
when fewer than `rounds` finish. Cells with no completed rounds stay NaN.
`stop_requested` records a stop request, not data completeness. Acquisition
exceptions propagate from Run and leave the previous successful record
unchanged. Save counts and cfg before closing the source. Load restores the
saved source without changing the current Form. Load rejects files with a
round axis (wrong number of axes) or older raw-IQ files without `sums`.
It does not convert them.

Analyze divides each cell's terminal gg count by that cell's total accepted
count, including other. Zero accepted shots give NaN survival; a finite
count cell from fewer rounds remains usable if the total is positive.
Malformed or partially populated cells cause an error. The fit uses only
seeds with finite survival in both arms at every requested depth. It weights
these seeds equally and requires at least two paired seeds and four distinct
depths.

## Interpreting the result

Analyze reports `estimate`, `fit_quality` and `reason`. Per-depth survival
and fitted curves stay inside analysis for plotting.

Paired decay analysis fits each arm to `A*p**depth+B`.
The two-qubit dimension is 4. The reported quantities are
`reference_epc = 3/4*(1-p_reference)` and
`gate_error = 3/4*(1-p_interleaved/p_reference)`.
Negative error estimates remain negative. They are diagnostics of the ratio
and its assumptions. Interpret a negative estimate as a model diagnostic,
while retaining the reported value.

The 95% interval resamples entire paired seed curves, preserving depth and arm
correlations. It describes statistical seed variation; independent-shot
uncertainty requires a different model. The summary reports the number of
paired seeds and valid bootstrap fits. The default is 1,000 deterministic resamples with bootstrap seed 42.

The live and pooled-survival plots center each bar on the count-weighted point.
The fit plot centers each bar on the equal-seed mean of the complete pairs.
Each bar extends one sample standard deviation (`ddof=1`) across seeds on either
side at that depth. A point with only one seed has no bar. The spread includes
sequence-to-sequence variation and shot noise. For survival probability `P` and
`N` shots contributing to a seed's estimate, shot noise is about `√(P(1−P)/N)`.
It grows as `P` falls from near one toward one-half, so a bar that grows with
depth does not by itself identify coherent error.

For each arm,
read `fit_quality.relative_parameter_errors["decay_time"]`. This number is
the fitted 1/e length's standard error divided by that length. A large
relative error can mean that the deepest point does not reach the tail and
the fit cannot separate the decay length from the offset. Add deeper points
and check the relative error again.

For example, with 11 synthetic depths, p = 0.98 and noise σ = 0.01, maximum
depths 30, 50, 80 and 200 gave relative errors 0.30, 0.14, 0.08 and 0.035.
R² was 0.993, 0.997, 0.998 and 0.999, so even the shortest sweep looked close
to its fitted curve. See the [comparison record](../../../../docs/discussions/2026-11-20-rb-round-reduction-and-fit-quality.md).
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

Calibrate each role's Xπ/2 and Xπ slots with amplitude and length Rabi before using them.
Confirm pulse frequency in MHz, signed gain in [-1, 1], and duration in us.
Use lookback to set mux timing, then mux onetone to set frequencies and gains.
Confirm readout timing in us and gains in [-1, 1].
Locate the dual-line CZ pulse with conditional spectroscopy and chevron.
Confirm the coupler's `cz_f1` and `cz_f2` slots before copying them into the Form.
