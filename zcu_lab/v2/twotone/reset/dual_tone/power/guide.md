## Behavior

Dual-tone reset gain map: a 2D sweep of the two reset-tone gains (pulse1 × pulse2 of the tested two-pulse reset) at the calibrated sideband frequencies, imaging the residual excitation to pick the gain pair that resets most completely. Runs on real hardware.

## Expects (parameter context)

Requires $qubit.ext.reset_f1 and $qubit.ext.reset_f2 for the two tested-reset frequencies. Missing, null or invalid values fail cfg lowering. Other drive defaults use captured qubit frequency and wiring.

## Expects (ModuleLibrary)

Builds the tested two-pulse-reset cfg from captured defaults and needs a readout module. Optionally references an upstream reset and an init pulse (a library pi pulse when present) — both disabled when no library entry exists.

## Typical writeback

Proposes the best gains to $qubit.ext.reset_gain1 and $qubit.ext.reset_gain2. With Run cfg and capture, proposes $qubit.reset.reset_120 from tested_reset, replacing only the two gains with this analysis. Tested frequencies and all other fields remain. An optional $qubit.module.reset row selects it.

## Recommended

Sweep each gain across its full usable range. Analysis denoises the map before picking the optimum; wavelet smoothing is the default and Gaussian remains available for comparison. The frequencies are held at 'reset_f1' / 'reset_f2', so calibrate those first.
