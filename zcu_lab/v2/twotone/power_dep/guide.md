## Behavior

Two-tone qubit power dependence: a 2D scan sweeping the qubit-drive gain (outer) against the drive frequency (inner), reading out the resonator at each point, to map how the qubit line shifts and broadens with drive power (AC-Stark shift, power broadening, multi-photon lines). Runs on real hardware. No automated fit — read off the 2D map by eye.

## Expects (parameter context)

Inline templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us from the fixed capture. Missing required calibration remains Invalid; no numeric fallback. Frequency sweep is qubit.freq ± 1.5 * qubit.kappa; gain runs 0.001 to 1.0.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a fresh readout template when unset or dangling. Optional qubit.module.reset stays disabled without a source. Library references remain selectable; sweeps overwrite their controlled fields only at Run. Qubit probe starts inline at template gain/frequency; both are overwritten by the sweeps.

## Typical writeback

No analysis and no writeback (the underlying experiment raises NotImplementedError). Inspect the 2D map manually; pick the low-power qubit frequency to feed back into 'qubit.freq' yourself.

## Recommended

No fit options. Default sweep: gain ~0.001→1.0 over ~101 points, frequency ~201 points spanning 1.5 linewidths around 'qubit.freq'. Start the gain sweep low to find the unsaturated line, then extend the upper bound to watch the AC-Stark shift and power broadening; widen the frequency span if higher-power features fall outside the window.
