## Behavior

**Readers:** operator, agent. Two-tone qubit flux dependence: a 2D scan stepping an external flux device (outer) against the qubit-drive frequency (inner), reading out the resonator at each point, to map the qubit spectrum versus flux (the Fluxonium flux-dispersion arcs). Runs on real hardware and needs a connected flux device. Extract points interactively from the GUI map. Run only after readout and qubit spectroscopy are credible.

**Last updated:** 2026-10-21

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Parameter sources

Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.time_of_fly + 0.05 us from the sources captured at Open/Reset. Missing required calibration leaves the fields Invalid until you supply calibration or direct values. Open and Reset leave both frequency endpoints blank and required, with 1001 points. Enter a reviewed drive-frequency window before Run. Flux spans 10% beyond qubit.flux_int and qubit.flux_half over 101 points; missing calibration leaves its endpoints Invalid. Both sweeps remain editable. Existing edits persist until Reset restores these starting values.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a fresh readout template when the slot is unset or points to a missing module. Optional qubit.module.reset stays disabled when absent. Saved modules remain selectable; sweeps replace only the swept fields at Run. Open/Reset fill flux_dev from the captured `$qubit.flux_source`; the direct name remains editable, and a missing source leaves it empty and Invalid. Choose flux_unit explicitly in the connected device's units; values keep that unit throughout the run.

## Typical writeback

After the run, the user drags two lines on the 2D map to mark the half-flux and integer-flux sweet spots, then clicks Done. The result writes back 'qubit.flux_half', 'qubit.flux_int' and 'qubit.flux_period' (= 2·|flx_int − flx_half|) as a calibration preview. Picking relies on user/agent judgement from the measured map. Apply a preview that replaces the earlier onetone/flux_dep calibration only after confirming the map is trustworthy.

## Recommended

Use after onetone/flux_dep has established the flux period and twotone/freq has found 'qubit.freq'. Calibrated readout and a verified qubit-drive window are prerequisites; otherwise the arc is usually invisible. Default sweep: flux has 101 points from 1.1 * qubit.flux_int − 0.1 * qubit.flux_half to 1.1 * qubit.flux_half − 0.1 * qubit.flux_int. Frequency has 1001 points and blank endpoints for the operator to fill. For a later model survey use a wide flux range and a broad frequency window at moderate drive gain to capture the full arc, then narrow both. For consecutive flux sweeps, reverse the sweep direction (swap start/stop) on the next run so the flux source need not ramp back across the full range first. Inspect the 2D map with gui_tab_get_figure (subtab_id='run'). Confirm 'flux_dev' points at a connected device before running.

## MCP recipe

Use `twotone_flux_dep` with required `center_mhz`, `span_mhz` and `freq_points`
for the qubit-drive scan. The endpoints are center minus/plus half the full span.
Also supply `flux_device`, exact cached `flux_unit`, two distinct native
`flux_range` endpoints and `flux_points`. No coordinate conversion occurs.
The recipe reads the cached device snapshot but does not connect the device.
`target` overrides the qubit alias for this Reset. `reuse_tab_id` resets an idle
matching tab. `readout_ref` selects a captured readout, and `gain` overrides the
drive amplitude. Null keeps the Form values. `reps` and `rounds` are optional.

After Run, `analyze(interactive=True)` hands the operation to the flux picker.
Move the half/integer lines and finish with Done. The recipe then asks before
applying selected qubit flux calibration items. Skip writes nothing.
The report contains `flux_half`, `flux_integer`, `flux_period` and
`writeback_applied`. Expected analysis failure returns `analysis_failure`
without writeback. Save retains raw data and completed analysis artifacts.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Supply both sweep axes and confirm the cached native flux unit. Check qubit probe and readout settings.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, Reset to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, Reset and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `twotone_flux_dep` after inspecting the same Form errors.
   center_mhz, span_mhz, freq_points, flux_device, flux_unit, flux_range and flux_points are required. Both counts map to their sweep Expts. The interactive picker uses the existing GUI parameters.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
