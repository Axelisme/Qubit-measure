## Behavior

Bath-reset cavity frequency–gain map: a 2D sweep of the bath reset's cavity-tone frequency × gain, with an internal 4-point pi/2 tomography phase axis the experiment drives itself, imaging the residual coherence to locate the (freq, gain) pair that best dumps the excitation into the cavity bath. Runs on real hardware.

## Expects (parameter context)

Requires `$resonator.freq` and `$qubit.ext.rabi_freq` for the cavity sweep from resonator frequency minus 1.2 Rabi frequency to minus 0.8 Rabi frequency. Other drive defaults use captured qubit/resonator frequency and wiring. Missing, null or invalid seed values fail cfg lowering.

## Expects (ModuleLibrary)

Builds the tested bath-reset cfg with cavity, qubit and pi/2 tones from captured defaults and needs a readout module. Optionally references an upstream reset and an init pulse (a library pi pulse when present), both disabled when no library entry exists.

## Typical writeback

Proposes the best cavity gain and frequency to `$qubit.ext.reset_bath_gain` and `$qubit.ext.reset_bath_freq` only. BathPhase proposes the tested ground/excited reset modules; length analysis has no automatic writeback.

## Recommended

Analysis denoises the 2D map before picking the peak; wavelet smoothing is the default and Gaussian remains available for comparison. Save writes one 3D HDF5 file with the tomography phase stored as an internal sweep axis.
