## Behavior

Dual-tone reset length sweep: holds the tested two-pulse reset at its calibrated sideband frequencies and gains and sweeps its duration (both tones together), showing how the residual excitation decays with reset length. Runs on real hardware. Run last, to pick the shortest length that fully resets the qubit.

## Expects (parameter context)

Requires $qubit.ext.reset_f1/reset_f2 and $qubit.ext.reset_gain1/reset_gain2 for the tested frequencies and gains. Missing, null or invalid values fail cfg lowering. Other drive defaults use captured qubit frequency and wiring.

## Expects (ModuleLibrary)

Builds the tested two-pulse-reset cfg from captured defaults and needs a readout module. Optionally references an upstream reset and an init pulse (a library pi pulse when present) — both disabled when no library entry exists.

## Typical writeback

No automatic writeback. Analysis plots the length response without fitting a length. Choose the length manually; DualTonePower proposes the tested reset module.

## Recommended

A length sweep from ~0.05 us to a few times the expected reset time captures the full decay; shorten the span once the plateau is clear.
