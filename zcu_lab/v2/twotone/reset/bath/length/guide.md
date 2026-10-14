## Behavior

Bath-reset length sweep: holds the tested bath reset at its calibrated cavity frequency and gain and sweeps its duration (cavity and qubit tones together), with an internal 4-point pi/2 tomography phase axis, showing how the residual excitation decays with reset length. Runs on real hardware. Run after the cavity freq/gain are found, to pick the shortest fully-resetting length.

## Expects (parameter context)

Requires `$qubit.ext.reset_bath_freq` and `$qubit.ext.reset_bath_gain` for the tested cavity tone. Missing, null or invalid values fail cfg lowering. Other drive defaults use captured qubit/resonator frequency and wiring.

## Expects (ModuleLibrary)

Builds the tested bath-reset cfg from captured defaults and needs a readout module. Optionally references an upstream reset and an init pulse (a library pi pulse when present) — both disabled when no library entry exists.

## Typical writeback

No writeback — the chosen reset length is read off the decay curve by eye, and registering the calibrated 'reset_bath' module is left to the final phase step.

## Recommended

A length sweep from ~0.05 us to a few times the expected reset time captures the full decay; shorten the span once the plateau is clear. Allow a long relax delay so the qubit fully relaxes before each reset.
