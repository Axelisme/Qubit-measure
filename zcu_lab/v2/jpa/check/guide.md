## Behavior

**Readers:** operator, agent. Run the JPA pump off/on resonator check from the GUI or MCP and compare the traces. The run sweeps the readout frequency with the selected JPA RF output off, then repeats with it on. Compare the two one-tone resonator traces by eye. This run commands real hardware. WARNING: the JPA pump output remains ON after the run. Review the selected RF device and sweep range before running. Treat the seeded bounds as bring-up defaults only, and establish safe limits for the connected hardware.

## Parameter sources

Uses the resonator alias for `$resonator.freq`, `$resonator.kappa` and resonator wiring, and the jpa (`amplifier/jpa`) alias for `$jpa.rf_source`. The qubit alias supplies an optional reset module. Opening or resetting the form reads the RF default from `$jpa.rf_source`; a missing source leaves the field empty and Invalid. The direct device name remains editable. Select the connected JPA RF device in the form. The check uses the existing pump-frequency, pump-power and flux settings on the hardware. Missing required resonator calibration marks the form Invalid.

## Module slots

Readout starts from the readout template. Its pulse uses `$resonator.freq` and `$resonator.wiring.ch`; acquisition uses `$resonator.freq`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.05. The sweep sets both frequencies during the run. You can also select a saved readout module. Optional reset uses `$qubit.module.reset`; an empty slot disables reset.

## Typical writeback

Use this visual diagnostic to compare the two traces. Update stored parameters in a separate step.

## Recommended

The Analyze tab shows the pump off/on comparison figure with fixed settings. A typical readout-frequency sweep uses about 101 points spanning 1.5 linewidths around `$resonator.freq`. Before every run, confirm the current JPA pump state and review the selected RF device. The run toggles its output off, then on, and leaves it ON.

**Last updated:** 2026-10-21
