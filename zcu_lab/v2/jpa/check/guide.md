## Behavior

JPA pump off/on resonator check: sweeps the readout frequency with the JPA pump RF output off and again with it on, recording the one-tone resonator response in both states so you can compare the resonance by eye. Runs on real hardware and commands the selected JPA RF device (its output is toggled off, then on). WARNING: the run leaves the JPA pump output ON when it finishes — the pump is NOT turned off afterwards. Review the selected RF device and the sweep range before running — the seeded bounds are bring-up defaults, not certified safety limits.

## Expects (parameter context)

Uses current aliases: qubit (qubit/*), resonator (resonator), and jpa (amplifier/jpa). Add a jpa alias row in the global dialog, for example $resonator.amplifier. Scalars use $jpa.pump_freq, $jpa.pump_power, $jpa.flux and canonical qubit/resonator wiring. Missing aliases or calibration leave cfg Invalid; no static fallback.

## Expects (ModuleLibrary)

SMART modules use $qubit.module.x180, $qubit.module.readout and optional $qubit.module.reset. Inline readout uses resonator scalar and wiring calibration; module waveforms follow their module source.

## Typical writeback

No writeback — the check is a visual diagnostic: compare the two traces and update parameters in another step.

## Recommended

No analysis options — the Analyze tab shows the pump off/on comparison figure only. Typical sweep: readout frequency ~101 points spanning 1.5 linewidths around $resonator.freq. Before every run, confirm the current JPA pump state and review the selected RF device: the run toggles its output off then on and leaves it ON.
