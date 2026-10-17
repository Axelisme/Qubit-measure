## Behavior

Hardware gain and three-branch sweep: Rabi / Rabi + reset / Rabi + reset + Rabi. Both pulses share the swept gain. Saves classified G/E populations; live view and analysis show Ground / Excited / Other with color for state and solid / dashed / dotted lines for branch.

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay is 5 * `$qubit.t1`; the gain sweep ends at 4 * `$qubit.module.x180.gain.` Classification seeds `$qubit.ext.readout_cal.g_center`/e_center/radius; missing calibration stays Invalid. Inline templates read `$qubit.freq`, `$qubit.wiring.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Other missing required sources remain Invalid.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Rabi uses `$qubit.module.x180` or pi_pulse. Tested reset uses `$qubit.module.reset` or reset template. Upstream reset is optional; readout uses `$qubit.module.readout` or readout template. Sweep-overwritten pulse values remain editable and library keys stay selectable. Run uses this fixed capture.

## Typical writeback

No writeback. Reports reset-only mean and worst sampled excited population and maximum Other fraction. No raw-IQ readout refit or histogram confidence intervals.

## Recommended

Reps are shots per gain per branch per round; rounds average repeated hardware sweeps and update the live plot. Branches are independent preparations. Other is not calibrated leakage; population estimates are not reset-channel fidelity.
