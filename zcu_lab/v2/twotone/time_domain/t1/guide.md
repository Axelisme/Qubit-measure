## Behavior

T1 energy relaxation: applies a pi pulse to excite the qubit, then sweeps a wait delay before readout and fits the exponential decay to extract T1. Runs on real hardware. Run after a pi pulse has been calibrated (amplitude/length Rabi).

## Expects (parameter context)

Delay runs 0 to 5 * qubit.t1 us over 101 points. Relax delay is 5 * qubit.t1 us. Missing calibration leaves the affected fields empty and Invalid; no numeric fallback.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a readout template fallback, and qubit.module.reset with no fallback (disabled when absent). The drive uses qubit.module.x180 with a pi_pulse template fallback. Unset or dangling slots use the named template, not a legacy blank. Key choices remain available. Inline templates read qubit.freq, qubit.wiring.L01.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Proposes fitted qubit.t1 and its standard error (us) to the captured qubit and working point. Also offers an independent sample with t1_us/t1_err_us at the Run bias when captured device evidence is available. No module writeback.

## Recommended

Analysis defaults to a single-exponential fit; enable dual-exponential only when the decay clearly shows two timescales (a fast component on top of the slow relaxation). A delay sweep reaching ~5*T1 lets the decay flatten so the fit is well-constrained. Supply a delay endpoint when qubit.t1 is missing. Set 'uniform=False' to cluster more points on the exponential decay while keeping the configured start/stop window.
