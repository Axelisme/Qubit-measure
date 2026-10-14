## Behavior

JPA flux × readout-frequency 2D survey: sweeps the JPA flux device value (outer axis) against the readout frequency (inner axis) and records the one-tone resonator response, so you can see by eye how the JPA response changes with flux and frequency. There is no automatic optimum analysis — judge the 2D map directly. Runs on real hardware and commands the selected JPA flux device. WARNING: review the selected flux device and both sweep ranges before running — the seeded bounds are bring-up defaults, not certified safety limits. The JPA pump is NOT controlled by this experiment: if the pump must be on for the measurement, confirm its state yourself (in the Devices interface) before running.

## Expects (parameter context)

Uses current aliases: qubit (qubit/*), resonator (resonator), and jpa (amplifier/jpa). Add a jpa alias row in the global dialog, for example $resonator.amplifier. Scalars use $jpa.pump_freq, $jpa.pump_power, $jpa.flux and canonical qubit/resonator wiring. Missing aliases or calibration leave cfg Invalid; no static fallback.

## Expects (ModuleLibrary)

SMART modules use $qubit.module.x180, $qubit.module.readout and optional $qubit.module.reset. Inline readout uses resonator scalar and wiring calibration; module waveforms follow their module source.

## Typical writeback

No writeback — this adapter has no analysis step (the underlying experiment has no optimum analysis yet). It produces a 2D map for visual inspection only; read off useful flux / frequency settings by eye and update parameters in another step.

## Recommended

No analysis options. Typical sweep: JPA flux device value ~101 points around the seeded centre ($jpa.flux), readout frequency ~101 points spanning 1.5 linewidths around 'r_f'. Confirm the JPA pump state yourself before every run. Survey the map wide first, then narrow the ranges around the response you want to inspect.
