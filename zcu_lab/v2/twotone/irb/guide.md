## Behavior

**Readers:** operator, agent. Paired single-qubit IRB: reference and interleaved arms share random Cliffords, each with its own full inverse. Depth counts random Cliffords; interleaved inserts one target per Clifford. Hardware sweeps depth and repetitions; software visits rounds, seeds and alternating arm order. Raw IQ has fixed axes round/seed/arm/depth, arm 0=reference, 1=interleaved.

**Last updated:** 2026-10-21

## Parameter sources

Depth starts in Log mode with max=128 and points=9, seed 42 and 20 paired sequences. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a readout template. Reset uses qubit.module.reset and stays disabled when absent. Drive slots use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. Unset slots or slots pointing to missing modules use the named template. Saved modules remain selectable. Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Analysis reports results for review; calibration stays unchanged. Fits both A*p**depth+B curves on a common IQ projection. Reports target error=(1-p_interleaved/p_reference)/2 and fidelity. 95% paired-seed bootstrap is statistical only; coherent or gate-dependent noise can bias this estimate. Estimates outside the physical range remain visible for review.

## Depth input

The depth input is a nonempty ordered list of unique nonnegative integers.
New Log forms produce [0,1,2,4,8,16,32,64,128]. Points includes zero and
must be at least 2. Log always includes 0 and max; two points yields [0,max].
Log rounds and deduplicates geometric samples. Effective points shows the actual
count when rounding collapses samples. Switch to List to edit the entire list.
Switching back fills max and points from the list and may resample it.
When loading older data, the stored IQ and depth axis remain readable. Re-enter depths in Log or List mode to rerun the experiment.

## Recommended

Default target X90. Use at least 4 distinct depths spanning decay and 20 or more seeds for uncertainty. Start with a short execution check. Analysis uses complete round/seed pairs; unfinished raw IQ is still saved.
