## Behavior

JPA pump off/on resonator check: sweeps the readout frequency with the JPA pump RF output off and again with it on, recording the one-tone resonator response in both states so you can compare the resonance by eye. Runs on real hardware and commands the selected JPA RF device (its output is toggled off, then on). WARNING: the run leaves the JPA pump output ON when it finishes — the pump is NOT turned off afterwards. Review the selected RF device and the sweep range before running — the seeded bounds are bring-up defaults, not certified safety limits.

## Expects (parameter context)

Uses the resonator alias for `$resonator.freq`, `$resonator.kappa` and resonator wiring. The qubit alias supplies an optional reset module. Select the connected JPA RF device in the form. RF defaults capture jpa_rf_dev at Open or Reset; an unbound alias leaves the optional field empty. This check does not use jpa pump-frequency, pump-power or flux calibration. Missing required resonator calibration leaves the form Invalid.

## Expects (ModuleLibrary)

Readout starts as a fresh readout template, not a library copy. Its inline pulse uses `$resonator.freq` and `$resonator.wiring.ch`; acquisition uses `$resonator.freq`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05. Run replaces both frequencies from the sweep. Readout library keys remain selectable. Optional reset uses `$qubit.module.reset`; an unset slot disables reset.

## Typical writeback

No writeback — the check is a visual diagnostic: compare the two traces and update parameters in another step.

## Recommended

No analysis options — the Analyze tab shows the pump off/on comparison figure only. Typical sweep: readout frequency ~101 points spanning 1.5 linewidths around `$resonator.freq`. Before every run, confirm the current JPA pump state and review the selected RF device: the run toggles its output off then on and leaves it ON.
