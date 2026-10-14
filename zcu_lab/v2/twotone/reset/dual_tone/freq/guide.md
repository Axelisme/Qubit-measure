## Behavior

Dual-tone reset frequency map: a 2D sweep of the two reset-tone frequencies (pulse1 × pulse2 of the tested two-pulse reset), imaging the qubit response to locate the pair of sideband frequencies that jointly dump the excitation. Runs on real hardware in hard-sweep mode (both axes are QICK register sweeps).

## Expects (parameter context)

Requires $qubit.ext.reset_f1/reset_f2 for the sweep centres and $qubit.ext.reset_gain1/reset_gain2 for the tested gains. Each frequency axis spans centre +/-50 MHz with 201 points. Missing, null or invalid values fail cfg lowering. Other drive defaults use captured qubit frequency and wiring.

## Expects (ModuleLibrary)

Builds the tested two-pulse-reset cfg from captured defaults and needs a readout module. Optionally references an upstream reset and an init pulse (a library pi pulse when present) — both disabled when no library entry exists.

## Typical writeback

Proposes the fitted frequencies to $qubit.ext.reset_f1 and $qubit.ext.reset_f2 only. DualTonePower proposes the tested reset module after analysing gains; length analysis has no automatic writeback.

## Recommended

Keep each axis a tight span (a few MHz) around the predicted sideband; the 2D scan cost grows with both axes. Analysis denoises the map before peak finding; wavelet smoothing is the default and Gaussian remains available for comparison.
