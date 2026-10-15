## Behavior

Bath-reset pi/2 phase sweep: holds the tested bath reset at its calibrated cavity freq/gain and length and sweeps the tomography pi/2 pulse phase, fitting the cosine response to find the phases that reset to ground (max) and to excited (min). Runs on real hardware. Run last, to fix the pi/2 phase of the reset module.

## Expects (parameter context)

Tested cavity frequency and gain read `$qubit.ext.reset_bath_freq` and `$qubit.ext.reset_bath_gain.` Seeds 201 phase points from -360 to 360 degrees, with relax delay 10.5 us. Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.L01.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Missing or invalid required sources leave the affected fields empty and Invalid; supply calibration or direct values before Run. No numeric calibration fallback is supplied.

## Expects (ModuleLibrary)

Tested reset starts inline from the bath_reset template. Optional init pulse uses `$qubit.module.x180` without fallback; an unset/dangling slot or unbound alias leaves it disabled and Valid. The pi/2 tone uses `$qubit.module.x90`, otherwise a fresh pi2_pulse template. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline readout template with a const waveform. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes phases to `$qubit.ext.reset_bath_max_phase` and `$qubit.ext.reset_bath_min_phase`. With Run cfg and capture, proposes `$qubit.reset.reset_bath` and `$qubit.reset.reset_bath_e` from tested_reset, replacing only pi/2 phase with max/ground and min/excited respectively. Tested cavity frequency, gain and all other fields remain. An optional `$qubit.module.reset` row selects ground by default; it can be edited to excited.

## Recommended

A full -360..360 deg sweep captures a clean cosine; the fit picks the max / min phases automatically. Allow a long relax delay so the qubit fully relaxes before each shot.
