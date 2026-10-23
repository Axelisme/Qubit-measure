## Behavior

**Readers:** operator, agent. Run a 2D JPA flux and readout-frequency survey from the GUI or MCP and inspect the map. The outer axis sweeps the JPA flux device value, and the inner axis sweeps readout frequency. Inspect the one-tone resonator response map by eye to choose useful settings. WARNING: this run commands the selected flux device on real hardware. Review that device and both sweep ranges before running. Treat the seeded bounds as bring-up defaults only, and establish safe limits for the connected hardware. The pump state stays under your control. If the measurement requires the pump on, confirm its state in the Devices interface before running.

## Parameter sources

Uses the resonator and jpa (`amplifier/jpa`) aliases. Bounds use `$resonator.freq`, `$resonator.kappa` and `$jpa.flux`. Opening or resetting the form selects the required JPA flux device from `flux_dev`; an unbound alias marks the form Invalid. `flux_unit` starts as A, and the device receives the entered values unchanged. The qubit alias supplies an optional reset module.

## Module slots

Readout starts from the readout template. Its pulse uses `$resonator.freq` and `$resonator.wiring.ch`; acquisition uses `$resonator.freq`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.05. The sweep sets both frequencies during the run. You can also select a saved readout module. Optional reset uses `$qubit.module.reset`; an empty slot disables reset.

## Typical writeback

The result is a 2D map for visual inspection. Choose useful flux and frequency settings by eye, then update stored parameters in a separate step.

## Recommended

A typical sweep uses about 101 JPA flux points around `$jpa.flux` and about 101 readout-frequency points spanning 1.5 linewidths around `$resonator.freq`. Confirm the JPA pump state before every run. Survey a wide map first, then narrow the ranges around the response you want to inspect.

**Last updated:** 2026-10-21
