## Behavior

Single-tone sideband-reset frequency sweep: drives the tested pulse-reset tone and sweeps its frequency, fitting the qubit response (Lorentzian) to find the reset sideband frequency and its linewidth. Runs on real hardware. Typically run after an init pulse excites the qubit, to locate the f01+f12 (`$resonator.freq` - `$qubit.freq`) sideband at which the reset pulse dumps the excitation.

## Expects (parameter context)

Seeds 201 frequency points at `$qubit.ext.reset_f` +/- 1.5 * `$qubit.ext.reset_fwhm.` The tested pulse has gain 0.3, length 5.0 us, and post-delay 5.0 / (2 * pi * `$resonator.kappa`) us. Relax delay reads `$qubit.t1.` Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Missing or invalid required sources leave the affected fields empty and Invalid; supply calibration or direct values before Run. No numeric calibration fallback is supplied.

## Expects (ModuleLibrary)

Tested reset starts inline from the reset template. Optional init pulse uses `$qubit.module.x180` without fallback; an unset/dangling slot or unbound alias leaves it disabled and Valid. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline readout template with a const waveform. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes fitted frequency and FWHM to `$qubit.ext.reset_f` and `$qubit.ext.reset_fwhm`. With Run cfg and capture, proposes `$qubit.reset.reset_10` from tested_reset, replacing only its frequency with this analysis. All other tested fields remain. An optional `$qubit.module.reset` row selects the concrete root.

## Recommended

A sweep of ~201 points spanning a couple of linewidths around the expected sideband captures the dip cleanly; widen the span if `$qubit.ext.reset_f` is only a rough resonator-minus-qubit estimate. The reset-pulse length is held fixed here — sweep it separately afterwards.
