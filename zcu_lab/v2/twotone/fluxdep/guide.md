## Behavior

**Readers:** operator, agent. Map two-tone qubit spectra versus DC bias, pick sweet spots and review period/offset writeback.

The outer sweep steps the external flux device; the inner sweep changes drive frequency and reads out the resonator. Hardware runs need a connected SoC and flux device. Confirm readout and qubit spectroscopy before mapping flux.

**Last updated:** 2026-10-21

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Parameter sources

Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.time_of_fly + 0.05 us from the sources captured at Open/Reset. Missing required calibration leaves the fields Invalid until you supply calibration or direct values. Open and Reset leave both frequency endpoints blank and required, with 1001 points. Enter a reviewed drive-frequency window before Run. Flux spans 10% beyond the selected device's actual integer and half over 101 points. The helper uses qubit.flux and other contributors' detached Form device values. Missing selected calibration leaves endpoints Invalid; invalid contributors raise an error. Both sweeps remain editable. Existing edits persist until Reset restores these starting values.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a fresh readout template when the slot is unset or points to a missing module. Optional qubit.module.reset stays disabled when absent. Saved modules remain selectable; sweeps replace only the swept fields at Run. Open/Reset fill flux_dev from the captured `$qubit.flux_dev`; the direct name remains editable, and a missing source leaves it empty and Invalid. Choose flux_unit explicitly in the connected device's units; values keep that unit throughout the run.

## Typical writeback

After the run, the user drags two lines on the 2D map to mark the half-flux and integer-flux sweet spots, then clicks Done. The preview separately proposes `flux.period.<cfg.flux_dev>` and a default-selected `flux.offset`. Period updates its magnitude and preserves its sign, initially positive.

Offset uses `0.5 - half_fit / period_s - Σ V_o / period_o`, with other values from the Run-start snapshot. The swept value is replaced by half_fit once. The note lists other present calibrated devices' values and contributions, including zero values. Missing device values contribute zero, and other periods stay unchanged. An ordinary record stores raw picked half/integer/positive period, scanned device, complete Run snapshot and original `{offset, period}` frame. Missing Run capture yields no proposals; missing the whole snapshot blocks calibration. Review period, offset and record items separately; period can be accepted without offset. Acceptance does not read live devices or current parameters. Picking relies on user/agent judgement from the measured map. Apply a preview that replaces the earlier onetone/flux_dep calibration only after confirming the map is trustworthy.

## Recommended

Use after onetone/flux_dep has established the flux period and twotone/freq has found 'qubit.freq'. Calibrated readout and a verified qubit-drive window are prerequisites; otherwise the arc is usually invisible. Default sweep: flux has 101 points between the selected device's actual integer and half, padded by 10% on each side. Picker initial lines project the Run-captured `{offset, period}` model onto that device with other devices' Run-start values. Missing other values contribute zero. Frequency has 1001 points and blank endpoints for the operator to fill. For a later model survey use a wide flux range and a broad frequency window at moderate drive gain to capture the full arc, then narrow both. For consecutive flux sweeps, reverse the sweep direction (swap start/stop) on the next run so the flux source need not ramp back across the full range first. Inspect the 2D map with gui_tab_get_figure (subtab_id='run'). Confirm 'flux_dev' points at a connected device before running.

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
