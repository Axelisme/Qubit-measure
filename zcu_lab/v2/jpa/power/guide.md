## Behavior

JPA pump-power calibration: with the qubit prepared in g and e (a pi pulse toggles it), sweeps the JPA pump power (dBm) and measures the g/e signal difference, so you can pick the pump power that best enhances readout. Runs on real hardware. WARNING: review the selected JPA RF device and the power sweep before running — the seeded bounds are bring-up defaults, not certified safety limits, and the run commands the selected device.

## Expects (parameter context)

Uses current aliases: qubit (qubit/*), resonator (resonator), and jpa (amplifier/jpa). Add a jpa alias row in the global dialog, for example $resonator.amplifier. Scalars use $jpa.pump_freq, $jpa.pump_power, $jpa.flux and canonical qubit/resonator wiring. Missing aliases or calibration leave cfg Invalid; no static fallback.

## Expects (ModuleLibrary)

SMART modules use $qubit.module.x180, $qubit.module.readout and optional $qubit.module.reset. Inline readout uses resonator scalar and wiring calibration; module waveforms follow their module source.

## Typical writeback

Proposes the signal-maximizing JPA pump power into parameter context 'best_jpa_power' (dBm) as a draft — it never writes it back without your acceptance, never commands a device from the writeback, and never touches 'cur_jpa_A'.

## Recommended

Review the selected RF device and the power sweep bounds before every run; the seeded range is a conservative low-power survey, only a starting point. Analysis picks the peak of the absolute signal difference.
