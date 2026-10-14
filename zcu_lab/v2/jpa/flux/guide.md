## Behavior

JPA flux-device calibration: with the qubit prepared in g and e (a pi pulse toggles it), sweeps the JPA flux device value and measures the g/e signal difference, so you can pick the flux device value that best enhances readout. Runs on real hardware. WARNING: review the selected JPA flux device and the sweep before running — the seeded bounds are bring-up defaults, not certified safety limits, and the run commands the selected device.

## Expects (parameter context)

Uses current aliases: qubit (qubit/*), resonator (resonator), and jpa (amplifier/jpa). Add a jpa alias row in the global dialog, for example $resonator.amplifier. Scalars use $jpa.pump_freq, $jpa.pump_power, $jpa.flux and canonical qubit/resonator wiring. Missing aliases or calibration leave cfg Invalid; no static fallback.

## Expects (ModuleLibrary)

SMART modules use $qubit.module.x180, $qubit.module.readout and optional $qubit.module.reset. Inline readout uses resonator scalar and wiring calibration; module waveforms follow their module source.

## Typical writeback

Proposes the signal-maximizing JPA flux device value into parameter context 'best_jpa_flux' as a draft — it never writes it back without your acceptance, never commands the device from the writeback, and never touches 'cur_jpa_A'.

## Recommended

Review the selected flux device and the sweep bounds before every run; the seeded range around the current centre is only a starting point. Analysis picks the peak of the absolute signal difference.
