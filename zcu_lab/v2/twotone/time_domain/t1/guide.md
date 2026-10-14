## Behavior

T1 energy relaxation: applies a pi pulse to excite the qubit, then sweeps a wait delay before readout and fits the exponential decay to extract T1. Runs on real hardware. Run after a pi pulse has been calibrated (amplitude/length Rabi).

## Expects (parameter context)

Requires a qubit and its referenced resonator. Defaults read qubit.t1 (us), qubit.freq (MHz), qubit.wiring.L01.ch, resonator.freq (MHz), and resonator.wiring.ch/ro_ch from the tab's fixed capture. Missing T1 seeds relax_delay at 100 us and the delay sweep stop at 500 us.

## Expects (ModuleLibrary)

Uses qubit.module.x180 and qubit.module.readout from the fixed capture. Unset or dangling slots seed inline pulse/readout defaults. Optional qubit.module.reset stays disabled when absent. Inline readout trigger offset defaults to 0.55 us; verify wiring and timing before running.

## Typical writeback

Proposes fitted qubit.t1 and its standard error (us) to the captured qubit and working point. Also offers an independent sample with t1_us/t1_err_us at the Run bias when captured device evidence is available. No module writeback.

## Recommended

Analysis defaults to a single-exponential fit; enable dual-exponential only when the decay clearly shows two timescales (a fast component on top of the slow relaxation). A delay sweep reaching ~5*T1 lets the decay flatten so the fit is well-constrained; with no prior qubit.t1 it spans 0–500 us. Set 'uniform=False' to cluster more points on the exponential decay while keeping the configured start/stop window.
