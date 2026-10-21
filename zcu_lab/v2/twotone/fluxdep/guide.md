## Behavior

**Readers:** operator, agent. Two-tone qubit flux dependence: a 2D scan stepping an external flux device (outer) against the qubit-drive frequency (inner), reading out the resonator at each point, to map the qubit spectrum versus flux (the Fluxonium flux-dispersion arcs). Runs on real hardware and needs a connected flux device. Extract points interactively from the GUI map. Run only after readout and qubit spectroscopy are credible.

**Last updated:** 2026-10-21

## Parameter sources

Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us from the sources captured at Open/Reset. Missing required calibration leaves the fields Invalid until you supply calibration or direct values. Open and Reset leave both frequency endpoints blank and required, with 1001 points. Enter a reviewed drive-frequency window before Run. Flux spans 10% beyond qubit.flux_int and qubit.flux_half over 101 points; missing calibration leaves its endpoints Invalid. Both sweeps remain editable. Existing edits persist until Reset restores these starting values.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a fresh readout template when the slot is unset or points to a missing module. Optional qubit.module.reset stays disabled when absent. Saved modules remain selectable; sweeps replace only the swept fields at Run. Open/Reset fill flux_dev with the device named by the captured flux_dev alias; a missing or unbound alias leaves it Invalid. Choose flux_unit explicitly in the connected device's units; values keep that unit throughout the run.

## Typical writeback

After the run, the user drags two lines on the 2D map to mark the half-flux and integer-flux sweet spots, then clicks Done. The result writes back 'qubit.flux_half', 'qubit.flux_int' and 'qubit.flux_period' (= 2·|flx_int − flx_half|) as a calibration preview. Picking relies on user/agent judgement from the measured map. Apply a preview that replaces the earlier onetone/flux_dep calibration only after confirming the map is trustworthy.

## Recommended

Use after onetone/flux_dep has established the flux period and twotone/freq has found 'qubit.freq'. Calibrated readout and a verified qubit-drive window are prerequisites; otherwise the arc is usually invisible. Default sweep: flux has 101 points from 1.1 * qubit.flux_int − 0.1 * qubit.flux_half to 1.1 * qubit.flux_half − 0.1 * qubit.flux_int. Frequency has 1001 points and blank endpoints for the operator to fill. For a later model survey use a wide flux range and a broad frequency window at moderate drive gain to capture the full arc, then narrow both. For consecutive flux sweeps, reverse the sweep direction (swap start/stop) on the next run so the flux source need not ramp back across the full range first. Inspect the 2D map with gui_tab_get_figure (subtab_id='run'). Confirm 'flux_dev' points at a connected device before running.
