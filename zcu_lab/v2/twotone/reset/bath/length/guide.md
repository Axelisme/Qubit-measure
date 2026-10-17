## Behavior

Bath-reset length sweep: holds the tested bath reset at its calibrated cavity frequency and gain and sweeps its duration (cavity and qubit tones together), with an internal 4-point pi/2 tomography phase axis, showing how the residual excitation decays with reset length. Runs on real hardware. Run after the cavity freq/gain are found, to pick the shortest fully-resetting length.

## Expects (parameter context)

Tested cavity frequency and gain read `$qubit.ext.reset_bath_freq` and `$qubit.ext.reset_bath_gain.` Seeds 201 length points from 0.05 to 15.0 us, with relax delay starting at 5*T1. Run replaces the pi/2 phase with its tomography cycle. Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Tested reset starts inline from the bath_reset template. Optional init pulse uses `$qubit.module.x180` without fallback; an unset/dangling slot or unbound alias leaves it disabled and Valid. The pi/2 tone uses `$qubit.module.x90`, otherwise a fresh pi2_pulse template. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline readout template with a const waveform. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

No writeback — the chosen reset length is read off the decay curve by eye, and registering the calibrated 'reset_bath' module is left to the final phase step.

## Recommended

A length sweep from ~0.05 us to a few times the expected reset time captures the full decay; shorten the span once the plateau is clear. Allow a long relax delay so the qubit fully relaxes before each reset.
