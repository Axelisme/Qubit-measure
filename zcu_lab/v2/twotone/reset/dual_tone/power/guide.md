## Behavior

Dual-tone reset gain map: a 2D sweep of the two reset-tone gains (pulse1 × pulse2 of the tested two-pulse reset) at the calibrated sideband frequencies, imaging the residual excitation to pick the gain pair that resets most completely. Runs on real hardware.

## Expects (parameter context)

Tested frequencies read `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2.` Seeds two 51-point gain axes from 0.0 to 1.0, with relax delay 0.5 us. Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Missing or invalid required sources leave the affected fields empty and Invalid; supply calibration or direct values before Run. No numeric calibration fallback is supplied.

## Expects (ModuleLibrary)

Tested reset starts inline from the two_pulse_reset template. Optional init pulse uses `$qubit.module.x180` without fallback; an unset/dangling slot or unbound alias leaves it disabled and Valid. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline readout template with a const waveform. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the best gains to `$qubit.ext.reset_gain1` and `$qubit.ext.reset_gain2`. With Run cfg and capture, proposes `$qubit.reset.reset_120` from tested_reset, replacing only the two gains with this analysis. Tested frequencies and all other fields remain. An optional `$qubit.module.reset` row selects it.

## Recommended

Sweep each gain across its full usable range. Analysis denoises the map before picking the optimum; wavelet smoothing is the default and Gaussian remains available for comparison. The frequencies are held at '`$qubit.ext.reset_f1`' / '`$qubit.ext.reset_f2`', so calibrate those first.
