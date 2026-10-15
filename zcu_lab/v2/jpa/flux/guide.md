## Behavior

JPA flux-device calibration: with the qubit prepared in g and e (a pi pulse toggles it), sweeps the JPA flux device value and measures the g/e signal difference, so you can pick the flux device value that best enhances readout. Runs on real hardware. WARNING: review the selected JPA flux device and the sweep before running — the seeded bounds are bring-up defaults, not certified safety limits, and the run commands the selected device.

## Expects (parameter context)

Uses the qubit (qubit/*), resonator and jpa (amplifier/jpa) aliases. The flux bounds use `$jpa.flux.` The optional JPA flux device resolves the flux_dev alias once at Open or Reset; an unbound alias leaves bias control disabled. flux_unit starts as A with no conversion. Missing required calibration leaves the form Invalid.

## Expects (ModuleLibrary)

Module slots use `$qubit.module.x180`, `$qubit.module.readout` and optional `$qubit.module.reset.` Missing pi/readout slots fall back to the pi_pulse/readout templates. Inline pulse calibration uses `$qubit.freq` and `$qubit.wiring.L01.ch`; inline readout uses resonator frequency/wiring and `$resonator.ext.time_of_fly` + 0.05. An unset reset slot disables reset. Missing required fallback calibration leaves the form Invalid.

## Typical writeback

Proposes the signal-maximizing JPA flux device value into `$jpa.flux`, in `$jpa.flux_unit` coordinates, as a draft. Nothing is written without your acceptance. Writeback never commands the device or changes the current-source value.

## Recommended

Review the selected flux device and the sweep bounds before every run; the seeded range around the current centre is only a starting point. Analysis picks the peak of the absolute signal difference.
