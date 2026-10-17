## Behavior

Bath-reset cavity frequency–gain map: a 2D sweep of the bath reset's cavity-tone frequency × gain, with an internal 4-point pi/2 tomography phase axis the experiment drives itself, imaging the residual coherence to locate the (freq, gain) pair that best dumps the excitation into the cavity bath. Runs on real hardware.

## Expects (parameter context)

Seeds 51 frequency points from `$resonator.freq` - 1.2 * `$qubit.ext.rabi_freq` to `$resonator.freq` - 0.8 * `$qubit.ext.rabi_freq`, and 51 gain points from 0.4 to 1.0. Relax delay reads 5.0 * `$qubit.t1.` The pi/2 phase offset stays locked at 90 degrees; Run adds its tomography phase cycle. Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Tested reset starts inline from the bath_reset template. Optional init pulse uses `$qubit.module.x180` without fallback; an unset/dangling slot or unbound alias leaves it disabled and Valid. The pi/2 tone uses `$qubit.module.x90`, otherwise a fresh pi2_pulse template. It is copied inline so the fixed 90-degree offset can be applied without changing the library. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline readout template with a const waveform. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the best cavity gain and frequency to `$qubit.ext.reset_bath_gain` and `$qubit.ext.reset_bath_freq` only. BathPhase proposes the tested ground/excited reset modules; length analysis has no automatic writeback.

## Recommended

Analysis denoises the 2D map before picking the peak; wavelet smoothing is the default and Gaussian remains available for comparison. Save writes one 3D HDF5 file with the tomography phase stored as an internal sweep axis.
