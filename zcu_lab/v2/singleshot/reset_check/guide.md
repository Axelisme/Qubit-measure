## Behavior

Hardware gain and three-branch sweep: Rabi / Rabi + reset / Rabi + reset + Rabi. Both pulses share the swept gain. Saves classified G/E populations; live view and analysis show Ground / Excited / Other with color for state and solid / dashed / dotted lines for branch.

## Expects (parameter context)

Freezes g_center / e_center / radius from resolved cfg. Set calibration directly or seed defaults with singleshot/ge writeback; invalid calibration fails before hardware. Optionally uses confusion_matrix for analysis correction. Reads pi_gain and t1 for defaults.

## Expects (ModuleLibrary)

Needs rabi_pulse, tested_reset and readout. Optional upstream reset prepares the initial state.

## Typical writeback

No writeback. Reports reset-only mean and worst sampled excited population and maximum Other fraction. No raw-IQ readout refit or histogram confidence intervals.

## Recommended

Reps are shots per gain per branch per round; rounds average repeated hardware sweeps and update the live plot. Branches are independent preparations. Other is not calibrated leakage; population estimates are not reset-channel fidelity.
