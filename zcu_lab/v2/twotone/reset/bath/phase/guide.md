## Behavior

**Readers:** operator, agent. Bath-reset pi/2 phase sweep: holds the tested bath reset at its calibrated cavity freq/gain and length and sweeps the tomography pi/2 pulse phase, fitting the cosine response to find the phases that reset to ground (max) and to excited (min). Runs on real hardware. Run last, to fix the pi/2 phase of the reset module.

**Last updated:** 2026-10-21

## Parameter sources

Tested cavity frequency and gain read `$qubit.ext.reset_bath_freq` and `$qubit.ext.reset_bath_gain`. Seeds 201 phase points from -360 to 360 degrees, with relax delay starting at 5*T1. Open and Reset fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the bath_reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. The pi/2 tone uses `$qubit.module.x90`, otherwise a fresh pi2_pulse template. Readout uses the captured `$qubit.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes phases to `$qubit.ext.reset_bath_max_phase` and `$qubit.ext.reset_bath_min_phase`. With acquired run settings and captured calibration, proposes `$qubit.reset.reset_bath` and `$qubit.reset.reset_bath_e` from the tested reset, replacing only pi/2 phase with max/ground and min/excited respectively. Tested cavity frequency, gain and all other fields remain. An optional `$qubit.module.reset` row selects ground by default; it can be edited to excited.

## Recommended

A full -360..360 deg sweep captures a clean cosine; the fit picks the max / min phases automatically. Allow a long relax delay so the qubit fully relaxes before each shot.
