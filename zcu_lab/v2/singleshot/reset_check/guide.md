## Behavior

**Readers:** operator, agent. How to compare single-shot reset branches from the GUI or MCP and interpret the results.

Hardware gain and three-branch sweep: Rabi / Rabi + reset / Rabi + reset + Rabi. Both pulses share the swept gain. Saves classified G/E populations; live view and analysis show Ground / Excited / Other with color for state and solid / dashed / dotted lines for branch.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay is 5 * `$qubit.t1`; the gain sweep ends at 4 * `$qubit.module.x180.gain`.

Classification starts from `$qubit.ext.readout_cal.g_center`, `e_center` and `radius`. Missing calibration leaves these fields Invalid.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Rabi uses `$qubit.module.x180` or pi_pulse. Tested reset uses `$qubit.module.reset` or reset template. Upstream reset is optional; readout uses `$qubit.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Reset snapshot.

## Typical writeback

This is a diagnostic only; calibration parameters stay unchanged. It reports the reset-only mean and worst sampled excited population and the maximum Other fraction. Saved data contain classified populations, so readout refitting requires a separate raw-IQ acquisition. Interpret these summaries as sampled populations rather than histogram confidence intervals.

## Recommended

Reps are shots per gain per branch per round; rounds average repeated hardware sweeps and update the live plot. Branches are independent preparations. Other is the unclassified remainder. Calibrated leakage and reset-channel fidelity require separate characterization.

**Last updated:** 2026-10-21
