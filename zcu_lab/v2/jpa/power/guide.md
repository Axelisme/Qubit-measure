## Behavior

JPA pump-power calibration: with the qubit prepared in g and e (a pi pulse toggles it), sweeps the JPA pump power (dBm) and measures the g/e signal difference, so you can pick the pump power that best enhances readout. Runs on real hardware. WARNING: review the selected JPA RF device and the power sweep before running — the seeded bounds are bring-up defaults, not certified safety limits, and the run commands the selected device.

## Expects (parameter context)

Uses the qubit (qubit/*), resonator and jpa (amplifier/jpa) aliases. The pump-power bounds use `$jpa.pump_power.` Select the connected JPA RF device in the form. RF defaults capture jpa_rf_dev at Open or Reset; an unbound alias leaves the optional field empty. Missing required aliases or calibration leave the form Invalid.

Relax delay starts at 1*T1 from the fixed capture. Valid T1 is a finite positive non-bool real number; missing/null T1 or absent capture uses 30 us. Stored invalid T1 leaves the wait required and empty, while source failures propagate. This editable initial value is not calibration writeback.

## Expects (ModuleLibrary)

Module slots use `$qubit.module.x180`, `$qubit.module.readout` and optional `$qubit.module.reset.` Missing pi/readout slots fall back to the pi_pulse/readout templates. Inline pulse calibration uses `$qubit.freq` and `$qubit.wiring.ch`; inline readout uses resonator frequency/wiring and `$resonator.ext.time_of_fly` + 0.05. An unset reset slot disables reset. Missing required fallback calibration leaves the form Invalid.

## Typical writeback

Proposes the signal-maximizing JPA pump power into `$jpa.pump_power` (dBm) as a draft. Nothing is written without your acceptance. Writeback never commands a device or changes the current-source value.

## Recommended

Review the selected RF device and the power sweep bounds before every run; the seeded range is a conservative low-power survey, only a starting point. Analysis picks the peak of the absolute signal difference.
