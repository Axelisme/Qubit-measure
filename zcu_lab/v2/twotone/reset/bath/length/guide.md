**Readers:** operator, agent. Measure how the bath-reset tomography response varies with qubit-tone duration. Apply optional preparation reset and preparation pulse, play both bath tones together, then play the tomography π/2 pulse after the cavity tone and read out.

**Last updated:** 2026-11-22

## Parameter sources

Tested cavity frequency and gain read `$qubit.ext.reset_bath_freq` and `$qubit.ext.reset_bath_gain`. Seeds 201 length points from 0.05 to 15.0 µs, with relax delay starting at 5*T1. Run replaces the pi/2 phase with its tomography cycle. Open and Defaults fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.015 µs for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Defaults capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the bath_reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. The pi/2 tone uses `$qubit.module.x90`, otherwise a fresh pi2_pulse template. Readout uses the captured `$resonator.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Acquisition and figures

Reps starts at 1000 per sweep point and branch in each acquisition round; Rounds starts at 10. Check the requested Reps × Rounds in Analyze before comparing runs. This length scan repeats the host sweep for each configured round, and each point within that sweep also acquires the configured number of rounds.

At each duration the final π/2 phase is set to 0°, 90°, 180° and 270° absolute. The qubit-tone duration is the x axis; cavity duration follows it with the difference in the source module held fixed. Opposite-phase tomography contrast is |S180−S0|² + |S270−S90|² in a.u.², with S equal to the complex mean at each phase. It measures phase sensitivity, not residual excitation or reset fidelity. The two IQ panels show mean I and mean Q for each labeled phase. Bars are the sample standard deviation of acquired shots in that quadrature divided by the square root of the retained count. The nonlinear contrast panel has no direct shot SEM. Mean-only files show no bars. If stopped rounds leave unequal shot counts for a point, its mean still gives each acquired host pass equal weight, and the plot omits its pooled SEM.

Choose a duration after the contrast settles; extend Stop if it is still changing at the last acquired point. Verify how that duration prepares the intended state before using it.

## Typical writeback

Read the chosen reset length from the contrast curve and enter it manually. Register the calibrated 'reset_bath' module in the final phase step.

## Recommended

Use a duration range that reaches beyond the first sustained plateau. Keep the preparation and readout the same while refining the range.
