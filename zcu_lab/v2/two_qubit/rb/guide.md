# Two-qubit paired RB and CZ-interleaved RB

**Readers:** operator, agent. Run paired two-qubit RB and CZ-interleaved RB through the GUI or MCP and interpret their error estimates.

**Last updated:** 2026-10-21

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Before run

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

Saved `$f1.ext.readout_cal` and `$f2.ext.readout_cal` g/e centers must match the
mux readout conditions. Raw acquisition is possible without centers, but FIT
then reports the missing saved calibration. Historical analysis uses only the
saved sources.

## Sequences and cost

`depths` is an increasing list of distinct nonnegative Clifford counts.
`seeds` is a list of distinct nonnegative 31-bit integers. Each integer is the
complete random-sequence identity. Keep these values with the data.
Each seed independently defines its random sequence. Reusing a
seed at shallower depths keeps the same random Clifford prefix.

Both arms draw from all 11,520 two-qubit Cliffords. Interleaved inserts one CZ
after each random Clifford. Reference already contains CZs inside the Clifford
decompositions. Each arm gets its own full-product inverse, including all
inserted CZs. Depth counts Cliffords; physical CZ and pulse counts depend on
the selected decompositions and arm.
Check representative compiled sequences for memory use and duration before a run.
A shallow depth can still contain several long CZ pulses.

Every depth/seed/arm starts with zero logical frames. Positive logical Z90
changes a later X90's programmed phase by minus 90 degrees. CZ playback
keeps its two charge phases fixed while logical frames evolve separately.
After CZ, the two named logical corrections advance the respective frames.
Virtual Z updates the logical frame only. Computational-basis readout is
invariant to terminal Z frames.

Each depth/seed/arm has one compiled program and `rounds` host acquisitions of
`reps` shots. The nominal B00 start leaves the thermal initial population in
place. Every shot retains the original thermal ensemble and proceeds directly
through the sequence to readout. The live plot counts completed shots in each arm.

## Save, cancel and analyze

Canonical raw IQ has axes `depth, seed, arm, round, shot, tone`. Arm 0 is
reference and arm 1 is CZ-interleaved. All original tones stay in their original
order. Round remains a separate axis. Saved IQ preserves same-shot correlations
and the readout's acquired ADC normalization.

Stop preserves completed whole rounds in their fixed slots. Unfinished rounds
stay complex NaN. Each completed acquisition occupies its assigned round slot
and contributes its shots once. An acquisition failure remains a failure even if
partial data exists. Save the raw data and cfg before closing the source.
Native Load restores that source for Analyze and leaves current Form input
unchanged.

Analyze classifies each saved tone against its saved g/e centers. Ties go to G.
It counts simultaneous G on both role tones in each shot to obtain joint00
survival directly from the paired role tones. Every thermal shot and its
original readout effects remain in the result. Missing samples remain missing.
Malformed or nonfinite acquired rounds cause an error.

Only round/seed pairs complete across both arms and every requested depth enter
the decay analysis. Analyze averages complete rounds within each seed and then
weights seeds equally. It requires at least two paired seeds and four distinct
depths. An incomplete run may meet those conditions, but it must retain the
same source identities and complete curves.

## Interpreting the result

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
Insufficient data, absent centers, flat decay, invalid fits or unstable bootstrap
stop estimation and retain the failure reason. Preserve that result and review
the data before deciding to remeasure. Statistical diagnostics and physical
acceptance remain operator decisions.

Thermal population, preparation and readout errors contribute to this ensemble.
Nearest-center binary classification leaves leakage unresolved. Finite shots
and too few shallow depths can leave the decay unidentifiable. A shallow
curve can also yield finite but biased parameters when amplitude and offset
are poorly constrained. Even a finite fit needs an identifiability review.
Gate-dependent noise, coherent errors and nonexponential decay can bias the
interleaved ratio beyond its statistical interval. Inspect the curves and those limitations before
interpreting a CZ error. This experiment provides diagnostic estimates only;
physical calibration acceptance is manual.
