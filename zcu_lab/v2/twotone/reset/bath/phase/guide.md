## Behavior

Bath-reset pi/2 phase sweep: holds the tested bath reset at its calibrated cavity freq/gain and length and sweeps the tomography pi/2 pulse phase, fitting the cosine response to find the phases that reset to ground (max) and to excited (min). Runs on real hardware. Run last, to fix the pi/2 phase of the reset module.

## Expects (parameter context)

Requires `$qubit.ext.reset_bath_freq` and `$qubit.ext.reset_bath_gain` for the tested cavity tone. Missing, null or invalid values fail cfg lowering. Other drive defaults use captured qubit/resonator frequency and wiring.

## Expects (ModuleLibrary)

Builds the tested bath-reset cfg from captured defaults and needs a readout module. Optionally references an upstream reset and an init pulse (a library pi pulse when present) — both disabled when no library entry exists.

## Typical writeback

Proposes phases to `$qubit.ext.reset_bath_max_phase` and `$qubit.ext.reset_bath_min_phase`. With Run cfg and capture, proposes `$qubit.reset.reset_bath` and `$qubit.reset.reset_bath_e` from tested_reset, replacing only pi/2 phase with max/ground and min/excited respectively. Tested cavity frequency, gain and all other fields remain. An optional `$qubit.module.reset` row selects ground by default; it can be edited to excited.

## Recommended

A full -360..360 deg sweep captures a clean cosine; the fit picks the max / min phases automatically. Allow a long relax delay so the qubit fully relaxes before each shot.
