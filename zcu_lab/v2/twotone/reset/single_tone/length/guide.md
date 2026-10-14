## Behavior

Single-tone sideband-reset length sweep: holds the tested pulse-reset at its calibrated sideband frequency and sweeps its duration, showing how the residual excitation decays with reset length. Runs on real hardware. Run after the reset frequency is found, to pick the shortest length that fully resets the qubit.

## Expects (parameter context)

Requires `$qubit.ext.reset_f` for the tested reset frequency. Missing, null or invalid values fail cfg lowering. Other drive defaults use captured qubit frequency and wiring.

## Expects (ModuleLibrary)

Builds the tested pulse reset inline and needs a readout module. Optionally references a calibrated upstream reset and an init pulse (a library pi pulse when present) — both disabled when no library entry exists.

## Typical writeback

No automatic writeback. Analysis plots the length response without fitting a length. Choose the length manually; SingleToneFreq proposes the tested reset module.

## Recommended

A length sweep from ~0.1 us to a few times the expected reset time captures the full decay; shorten the span once the plateau is clear. Allow a long relax delay between shots so the qubit fully relaxes before each reset.
