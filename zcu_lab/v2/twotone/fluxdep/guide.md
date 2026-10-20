## Behavior

Two-tone qubit flux dependence: a 2D scan stepping an external flux device (outer) against the qubit-drive frequency (inner), reading out the resonator at each point, to map the qubit spectrum versus flux (the Fluxonium flux-dispersion arcs). Runs on real hardware and needs a connected flux device. No automated fit in the GUI — points are extracted interactively. This is not the early flux-finder; use it only after readout and qubit spectroscopy are already credible.

## Expects (parameter context)

Inline templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us from the fixed capture. Missing required calibration remains Invalid; no numeric fallback. Open and Reset leave both frequency endpoints blank and required, with 1001 points. Enter a reviewed drive-frequency window before Run. Flux spans 10% beyond qubit.flux_int and qubit.flux_half over 101 points; missing calibration leaves its endpoints Invalid. Both sweeps remain editable. Existing edits persist until Reset restores these starting values.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a fresh readout template when unset or dangling. Optional qubit.module.reset stays disabled without a source. Library references remain selectable; sweeps overwrite their controlled fields only at Run. flux_dev resolves the captured flux_dev alias to a direct device name on Open/Reset; missing or unbound alias remains Invalid. flux_unit stays explicit and is never converted.

## Typical writeback

Interactive analysis (not a fit): after the run, the user drags two lines on the 2D map to mark the half-flux and integer-flux sweet spots, then clicks Done. The result writes back 'qubit.flux_half', 'qubit.flux_int' and 'qubit.flux_period' (= 2·|flx_int − flx_half|) to the parameter context as a preview. Picking is user/agent judgement from the measured map; do not use this preview to replace the earlier onetone/flux_dep flux calibration unless the map is clearly trustworthy.

## Recommended

No fit options. Use after onetone/flux_dep has established the flux period and twotone/freq has found 'qubit.freq'. Do not run this early to find flux: with uncalibrated readout or an unknown qubit-drive window the arc is usually invisible. Default sweep: flux has 101 points from 1.1 * qubit.flux_int − 0.1 * qubit.flux_half to 1.1 * qubit.flux_half − 0.1 * qubit.flux_int. Frequency has 1001 points and blank endpoints for the operator to fill. For a later model survey use a wide flux range and a broad frequency window at moderate drive gain to capture the full arc, then narrow both. For consecutive flux sweeps, reverse the sweep direction (swap start/stop) on the next run so the flux source need not ramp back across the full range first. Inspect the 2D map with gui_tab_get_figure (subtab_id='run'). Confirm 'flux_dev' points at a connected device before running.
