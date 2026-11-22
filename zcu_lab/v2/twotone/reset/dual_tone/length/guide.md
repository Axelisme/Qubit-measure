**Readers:** operator, agent. Measure the duration dependence of two simultaneous reset tones at calibrated RF frequencies and gains. The sequence is optional preparation reset → optional preparation pulse → both reset tones → readout after the longer tone.

**Last updated:** 2026-11-22

## Parameter sources

Tested frequencies read `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2`; gains read `$qubit.ext.reset_gain1` and `$qubit.ext.reset_gain2`. Seeds 51 length points from 0.05 to 40.0 µs, with relax delay starting at 5*T1. Open and Defaults fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.015 µs for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Defaults capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the two_pulse_reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. Readout uses the captured `$resonator.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Acquisition and figures

Reps starts at 1000 per sweep point and branch in each acquisition round; Rounds starts at 10. Check the requested Reps × Rounds in Analyze before comparing runs.

The horizontal coordinate is tone 1 duration. At each point, tone 2 uses tone 1 duration plus the difference between their durations in the form; set Start high enough that both pulses have positive duration. Live and Analyze show the mean IQ projected onto its principal variation axis, falling back to I for fewer than two available means. The projection is a readout contrast rather than a population. Each vertical bar is the per-point shot SEM: sample standard deviation along that same IQ axis divided by the square root of the retained shot count. It measures uncertainty of the mean, not shot scatter or fit-parameter uncertainty. Files saved before shot moments were recorded still load and show points without bars.

Choose the shortest sampled duration on a sustained plateau within the error bars. Extend Stop if the response is still changing at the endpoint. Once it levels off, refine Start and Stop around the first plateau transition and keep the configured tone duration difference. Check state preparation separately before applying that duration.

## Typical writeback

Analysis plots the length response for manual selection. Calibration stays unchanged until you update it. The dual-tone power experiment proposes the tested reset module.

## Recommended

Use a duration range that reaches beyond the first sustained plateau. Keep the preparation and readout the same while refining the range.
