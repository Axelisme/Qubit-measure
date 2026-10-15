## Behavior

Single-tone sideband-reset length sweep: holds the tested pulse-reset at its calibrated sideband frequency and sweeps its duration, showing how the residual excitation decays with reset length. Runs on real hardware. Run after the reset frequency is found, to pick the shortest length that fully resets the qubit.

## Expects (parameter context)

The tested pulse frequency reads `$qubit.ext.reset_f.` Seeds 50 length points from 0.1 to 20.0 us, with relax delay 30.5 us. Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.L01.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Missing or invalid required sources leave the affected fields empty and Invalid; supply calibration or direct values before Run. No numeric calibration fallback is supplied.

## Expects (ModuleLibrary)

Tested reset starts inline from the reset template. Optional init pulse uses `$qubit.module.x180` without fallback; an unset/dangling slot or unbound alias leaves it disabled and Valid. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline readout template with a const waveform. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

No automatic writeback. Analysis plots the length response without fitting a length. Choose the length manually; SingleToneFreq proposes the tested reset module.

## Recommended

A length sweep from ~0.1 us to a few times the expected reset time captures the full decay; shorten the span once the plateau is clear. Allow a long relax delay between shots so the qubit fully relaxes before each reset.
