**Readers:** operator, agent. Sweep the duration of a sideband reset at its calibrated carrier and fixed gain. Each shot applies the optional preparation reset and preparation pulse, then the tested sideband pulse, then readout.

**Last updated:** 2026-11-22

## Parameter sources

The tested pulse frequency reads `$qubit.ext.reset_f`. Seeds 50 length points from 0.1 to 20.0 µs, with relax delay starting at 5*T1. Open and Defaults fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.015 µs for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Defaults capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. Readout uses the captured `$resonator.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Acquisition and figures

Reps starts at 1000 per sweep point and branch in each acquisition round; Rounds starts at 10. Check the requested Reps × Rounds in Analyze before comparing runs.

Live and Analyze plot mean IQ projected onto the principal variation axis of the sweep (raw I if fewer than two means are available). This is not population or reset fidelity. Compare the change with duration, not the sign of the projected value. Each vertical bar is the per-point shot SEM: sample standard deviation along that same IQ axis divided by the square root of the retained shot count. It measures uncertainty of the mean, not shot scatter or fit-parameter uncertainty. Files saved before shot moments were recorded still load and show points without bars.

For manual duration selection, look for the earliest acquired duration at which the response reaches a plateau and stays there within the point errors. If the trace is still changing at the right edge, extend Stop; if it starts on the plateau, lower Start or use finer steps around its onset. Enter the chosen duration in the tested pulse or saved reset module. Confirm the state with a separate preparation/readout calibration before calling it fully reset.

## Typical writeback

Analysis plots the length response for manual selection. Calibration stays unchanged until you update it. The single-tone frequency experiment proposes the tested reset module.

## Recommended

Use a duration range that reaches beyond the first sustained plateau. Keep the preparation and readout the same while refining the range.
