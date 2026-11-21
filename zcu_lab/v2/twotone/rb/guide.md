# Single-qubit randomized benchmarking

**Readers:** operator, agent. Run reference RB or target-interleaved RB and read the decay fits.

**Last updated:** 2026-11-20, per-depth seed spread

## Choose a target

Open `twotone/rb`. Leave **Target gate** at None to measure the reference decay.
Select X90, X180, Y90 or Y180 to measure the reference and interleaved decays in
the same run. Both arms use the same random Clifford sequences. The interleaved
arm inserts the target after every random Clifford. Each arm applies its own
full inverse before readout. Depth counts random Cliffords, not physical pulses
or inserted targets.

The board sweeps depth and runs the repetitions for each seed and arm.
`rounds` is the number of acquisition rounds per program. Acquisition
averages IQ across `reps` and finished rounds. Each seed runs reference first,
then the optional interleaved arm. The live plot shows one line per arm.

## Set the starting values

Calibrate qubit frequency, the X90 and X180 pulse amplitudes, and readout before
Run. Open and Defaults take the X90 and X180 pulses from the qubit's module slots
and readout from the resonator's readout slot. If a slot is empty or points to a
missing module, the form uses the corresponding `pi2_pulse`, `pi_pulse` or
`readout` template. The templates use the captured frequency and wiring.
The readout template uses the resonator's time of flight plus 0.015 µs.
Missing required calibration leaves the affected input Invalid. You can select
a saved module instead. The reset module comes from the qubit's reset slot and
stays off when that slot is empty.

Open and Defaults set Relax delay to five times the captured qubit T1. If T1 or
the capture is absent, the starting value is 150 µs. An invalid stored T1
leaves the input Invalid. Active reset does not change this wait. Check the
modules, readout timing and relax delay before starting.

Depth starts in Log mode with max 128 and nine points:
`[0, 1, 2, 4, 8, 16, 32, 64, 128]`. Log includes zero and max, and rounds
and removes duplicate depths. Effective points displays the count after
deduplication. Use List mode to enter distinct nonnegative integer depths in
the order you want to run them. Switching back to Log may resample the list.

The form starts at seed 42 with 20 sequences. Start with a short run to check
execution. For analysis, use at least four distinct depths that span the decay
and aim for 20 or more independent seeds. Reuse the seed to reproduce the
random sequences.

## Read the result

Native data stores averaged IQ `signals` in (seed, arm, depth) order. Arm 0 is
reference. Arm 1 is present only when a target is selected. Stop keeps the
average of completed rounds for the current seed and arm. Cells without any
completed round stay NaN. Analyze keeps seeds with finite IQ in all arms
at every depth. It weights them equally and needs at least two complete seeds.

Analyze fits `A * p**depth + B` on one IQ projection and plots the observations
and fitted curves. Reference-only mode reports decay `p`, error per Clifford
`(1 - p) / 2` and its 95% seed-bootstrap interval. In paired mode it reports
both decays, reference error per Clifford and the target error estimate
`(1 - p_interleaved / p_reference) / 2`. The target fidelity is one minus that
error, with a 95% paired-seed bootstrap interval.

Each live and fit point shows one sample standard deviation (`ddof=1`) across
seeds on either side, measured at that depth on the same IQ axis as the point.
A point with only one seed has no bar. The spread combines sequence-to-sequence
variation and shot noise. For a survival probability `P` estimated from `N`
shots, the shot-noise scale is about `√(P(1−P)/N)`. It grows as `P` falls from
near one toward one-half. Growing spread with depth alone does not identify
coherent error.

These intervals describe seed variation. They do not account for bias from
gate-dependent or coherent noise. A negative target error or fidelity above one
remains visible. Review the curves and interval before using the estimate.
Analyze does not write back calibration.

Check that the deepest points approach the fitted offset before using the
decay estimate. This experiment does not report fit quality. The
[two-qubit RB guide](../../two_qubit/rb/guide.md#interpreting-the-result)
shows how a short scan can have R² of 0.993 while the decay length has 30%
relative error.

Load rejects older files with a round axis (wrong number of axes). It does
not convert these files or previous standalone RB and IRB runs. Re-enter the
settings and Run to save the current format.

## Reading the plots

Live shows an IQ projection by random Clifford depth; its title shows the number of complete seeds. The Analyze marks are the means across eligible seeds. Bars are across-seed sample standard deviation, not shot SEM, and do not give the uncertainty of the mean IQ over repetitions. The 95% bootstrap interval estimates variation when resampling finite seeds, not systematic model error. The summary shows configured Reps × Rounds for each arm and seed.
