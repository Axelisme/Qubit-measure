## Behavior

**Readers:** operator, agent. Two-tone qubit power dependence: a 2D scan sweeping the qubit-drive gain (outer) against the drive frequency (inner), reading out the resonator at each point, to map how the qubit line shifts and broadens with drive power (AC-Stark shift, power broadening, multi-photon lines). Runs on real hardware. Read the 2D map manually.

**Last updated:** 2026-10-21

## Parameter sources

Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us from the sources captured at Open/Reset. Missing required calibration leaves the fields Invalid until you supply calibration or direct values. Frequency sweep is qubit.freq ± 1.5 * qubit.kappa; gain runs 0.001 to 1.0.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a fresh readout template when the slot is unset or points to a missing module. Optional qubit.module.reset stays disabled when absent. Saved modules remain selectable; sweeps replace only the swept fields at Run. Qubit probe starts from a template's gain and frequency; both are overwritten by the sweeps.

## Typical writeback

Inspect the 2D map manually. Enter the observed low-power qubit frequency into 'qubit.freq' after review; calibration stays unchanged until you update it.

## Recommended

Default sweep: gain ~0.001→1.0 over ~101 points, frequency ~201 points spanning 1.5 linewidths around 'qubit.freq'. Start the gain sweep low to find the unsaturated line, then extend the upper bound to watch the AC-Stark shift and power broadening; widen the frequency span if higher-power features fall outside the window.
