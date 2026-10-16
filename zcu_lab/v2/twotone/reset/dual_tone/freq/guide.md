## Behavior

Dual-tone reset frequency map: a 2D sweep of the two reset-tone frequencies (pulse1 × pulse2 of the tested two-pulse reset), imaging the qubit response to locate the pair of sideband frequencies that jointly dump the excitation. Runs on real hardware in hard-sweep mode (both axes are QICK register sweeps).

## Expects (parameter context)

Seeds two 201-point axes at `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2`, each +/- 50 MHz. Tested gains read `$qubit.ext.reset_gain1` and `$qubit.ext.reset_gain2.` Relax delay is 0.5 us. Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Missing or invalid required sources leave the affected fields empty and Invalid; supply calibration or direct values before Run. No numeric calibration fallback is supplied.

## Expects (ModuleLibrary)

Tested reset starts inline from the two_pulse_reset template. Optional init pulse uses `$qubit.module.x180` without fallback; an unset/dangling slot or unbound alias leaves it disabled and Valid. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline readout template with a const waveform. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the fitted frequencies to `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2` only. DualTonePower proposes the tested reset module after analysing gains; length analysis has no automatic writeback.

## Recommended

Keep each axis a tight span (a few MHz) around the predicted sideband; the 2D scan cost grows with both axes. Analysis denoises the map before peak finding; wavelet smoothing is the default and Gaussian remains available for comparison.
