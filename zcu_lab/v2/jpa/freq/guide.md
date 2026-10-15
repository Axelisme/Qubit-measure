## Behavior

JPA pump-frequency tracer: with the qubit prepared in g and e (a pi pulse toggles it), sweeps the JPA pump frequency and measures the g/e signal difference, so you can pick the pump frequency that best enhances readout. Runs on real hardware. WARNING: review the selected JPA RF device and the pump sweep before running — the seeded bounds are bring-up defaults, not certified safety limits, and the run commands the selected device.

## Expects (parameter context)

Uses the qubit (qubit/*), resonator and jpa (amplifier/jpa) aliases. The pump-frequency bounds use `$jpa.pump_freq.` Select the connected JPA RF device in the form. RF defaults capture jpa_rf_dev at Open or Reset; an unbound alias leaves the optional field empty. Missing required aliases or calibration leave the form Invalid.

## Expects (ModuleLibrary)

Module slots use `$qubit.module.x180`, `$qubit.module.readout` and optional `$qubit.module.reset.` Missing pi/readout slots fall back to the pi_pulse/readout templates. Inline pulse calibration uses `$qubit.freq` and `$qubit.wiring.L01.ch`; inline readout uses resonator frequency/wiring and `$resonator.ext.time_of_fly` + 0.05. An unset reset slot disables reset. Missing required fallback calibration leaves the form Invalid.

## Typical writeback

Proposes the signal-maximizing JPA pump frequency into `$jpa.pump_freq` (MHz) as a draft. Nothing is written without your acceptance, and writeback never changes the current-source value.

## Recommended

Review the selected RF device and the sweep bounds before every run; the seeded range around `$jpa.pump_freq` is only a starting point. Analysis picks the peak of the absolute signal difference.
