## Behavior

**Readers:** operator, agent. T1 energy relaxation: applies a pi pulse to excite the qubit, then sweeps a wait delay before readout and fits the exponential decay to extract T1. Runs on real hardware. Run after a pi pulse has been calibrated (amplitude/length Rabi).

**Last updated:** 2026-10-21

## Parameter sources

Delay runs 0 to 5 * qubit.t1 us over 101 points. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

The ordinary T1 sweep stop is 5 times the same time scale, initially 150 us when T1 is missing.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a readout template. Reset uses qubit.module.reset and stays disabled when absent. The drive uses qubit.module.x180 with a pi_pulse template fallback. Unset slots or slots pointing to missing modules use the named template. Saved modules remain selectable. Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Proposes fitted qubit.t1 and its standard error (us) to the captured qubit and working point. Also offers an independent sample with t1_us/t1_err_us at the Run bias when captured device evidence is available. Module slots stay unchanged.

## Recommended

Analysis defaults to a single-exponential fit; enable dual-exponential only when the decay clearly shows two timescales (a fast component on top of the slow relaxation). A delay sweep reaching ~5*T1 lets the decay flatten so the fit is well-constrained. Missing T1 starts the delay endpoint at 150 us; stored invalid T1 leaves it required and empty. Set 'uniform=False' to cluster more points on the exponential decay while keeping the configured start/stop window.
