**Readers:** operator, agent. Find opposite tomography π/2 phases for a fixed bath reset. Apply optional preparation reset and preparation pulse, start the cavity and qubit tones together, play the phase-swept π/2 pulse after the cavity tone, then read out.

**Last updated:** 2026-11-22

## Parameter sources

Tested cavity frequency and gain read `$qubit.ext.reset_bath_freq` and `$qubit.ext.reset_bath_gain`. Seeds 201 phase points from -360 to 360 degrees, with relax delay starting at 5*T1. Open and Defaults fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.015 µs for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Defaults capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the bath_reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. The pi/2 tone uses `$qubit.module.x90`, otherwise a fresh pi2_pulse template. Readout uses the captured `$resonator.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Acquisition and figures

Reps starts at 1000 per sweep point and branch in each acquisition round; Rounds starts at 10. Check the requested Reps × Rounds in Analyze before comparing runs.

Live and Analyze show mean IQ projected onto the principal variation axis (raw I if fewer than two means are available). The ordinate is not calibrated population. Each vertical bar is the per-point shot SEM: sample standard deviation along that same IQ axis divided by the square root of the retained shot count. It measures uncertainty of the mean, not shot scatter or fit-parameter uncertainty. Files saved before shot moments were recorded still load and show points without bars.

Analyze overlays a fixed 360° cosine and marks its maximum and minimum, reported in the turn nearest zero. The dashed markers show equivalent phases nearest the middle of the measured sweep. Their ± values are 1σ phase errors from the fit covariance. For a preferred extremum, set the new π/2 phase to the annotated value modulo 360°; the change is that phase minus the phase in the source pulse, wrapped to the nearest turn. Use independently prepared references to verify which extremum corresponds to g or e. A nearly flat fringe or scan too short to resolve both extrema does not establish the state assignment.

## Typical writeback

Proposes phases to `$qubit.ext.reset_bath_max_phase` and `$qubit.ext.reset_bath_min_phase`. With acquired run settings and captured calibration, proposes `$qubit.reset.reset_bath` and `$qubit.reset.reset_bath_e` from the tested reset, replacing only pi/2 phase with the fitted maximum and minimum respectively; verify that the default ground/excited assignment matches calibrated state references. Tested cavity frequency, gain and all other fields remain. An optional `$qubit.module.reset` row selects ground by default; it can be edited to excited.

The proposal uses only this Run's tested reset and captured parameters, not current aliases, ext values or an older library module. Missing saved cfg or parameter capture leaves only the scalar proposals. Source and Apply errors still propagate. Applying the library rows without the optional reset-slot row leaves the selected reset unchanged.

## Recommended

A full -360..360 deg sweep captures a clean cosine; the fit picks the max / min phases automatically. Allow a long relax delay so the qubit fully relaxes before each shot.
