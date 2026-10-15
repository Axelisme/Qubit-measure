## Behavior

Two-tone qubit flux dependence: a 2D scan stepping an external flux device (outer) against the qubit-drive frequency (inner), reading out the resonator at each point, to map the qubit spectrum versus flux (the Fluxonium flux-dispersion arcs). Runs on real hardware and needs a connected flux device. No automated fit in the GUI — points are extracted interactively. This is not the early flux-finder; use it only after readout and qubit spectroscopy are already credible.

## Expects (parameter context)

Inline templates read qubit.freq, qubit.wiring.L01.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us from the fixed capture. Missing required calibration remains Invalid; no numeric fallback. Frequency sweep is qubit.freq ± qubit.kappa over 1001 points. Flux spans 10% beyond qubit.flux_int and qubit.flux_half; missing endpoints remain Invalid.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a fresh readout template when unset or dangling. Optional qubit.module.reset stays disabled without a source. Library references remain selectable; sweeps overwrite their controlled fields only at Run. flux_dev resolves the captured flux_dev alias to a direct device name on Open/Reset; missing or unbound alias remains Invalid. flux_unit stays explicit and is never converted.

## Typical writeback

Interactive analysis (not a fit): after the run, the user drags two lines on the 2D map to mark the half-flux and integer-flux sweet spots, then clicks Done. The result writes back 'qubit.flux_half', 'qubit.flux_int' and 'qubit.flux_period' (= 2·|flx_int − flx_half|) to the parameter context as a preview. Picking is user/agent judgement from the measured map; do not use this preview to replace the earlier onetone/flux_dep flux calibration unless the map is clearly trustworthy.

## Recommended

No fit options. Use after onetone/flux_dep has established the flux period and twotone/freq has found 'qubit.freq'. Do not run this early to find flux: with uncalibrated readout or an unknown qubit-drive window the arc is usually invisible. Default sweep: flux ~101 points across roughly one period (bracketed by 'qubit.flux_int'/'qubit.flux_half' when calibrated), frequency ~1001 points spanning ±qubit.kappa around 'qubit.freq'. For a later model survey use a wide flux range and a broad frequency window at moderate drive gain to capture the full arc, then narrow both. For consecutive flux sweeps, reverse the sweep direction (swap start/stop) on the next run so the flux source need not ramp back across the full range first. Inspect the 2D map with gui_tab_get_figure (subtab_id='run'). Confirm 'flux_dev' points at a connected device before running.
