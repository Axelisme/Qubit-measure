## Behavior

**Readers:** operator, agent. How to measure one-tone resonator flux dependence from the GUI or MCP and locate flux sweet spots.

The 2D sweep varies an external flux-bias device value and readout frequency to trace how the resonator frequency moves with flux. Use the map to locate a Fluxonium's half-flux and integer-flux sweet spots. The measurement runs on real hardware and requires a SoC connection and a configured flux-bias device.

**Last updated:** 2026-10-21

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Parameter sources

Opening a tab or pressing Reset takes a fixed copy of the current parameters and evaluates starting expressions once into editable input values. The frequency sweep starts with 101 points at `$resonator.freq` ± `$resonator.kappa`, in MHz. The readout uses `$resonator.freq`, `$resonator.wiring.ch`, and `$resonator.wiring.ro_ch` for its frequency and channels. The ADC window starts at 0.99 us. Trigger timing reads the top-level `$resonator.time_of_fly`, adding `5/(2π·kappa)` us when kappa is set. Missing or null TOF leaves trigger and pulse length required and blank. Pulse length starts at the larger of 1.0 us and trigger plus the ADC window.

The flux sweep starts with 101 points from 1.1 × `$qubit.flux_int` − 0.1 × `$qubit.flux_half` to 1.1 × `$qubit.flux_half` − 0.1 × `$qubit.flux_int`, in the selected device unit. Missing or invalid required sources leave the affected fields empty and Invalid, blocking the run until valid values are supplied. For first bring-up before calibration, enter safe, user-approved frequency and flux bounds.

The sweet-spot picker reads `$qubit.flux_half` and `$qubit.flux_int` from the selected Run or Load parameter source. Missing or null values use the picker's default line positions. Choose sweet spots from the measured map.

## Readout module

The readout starts as a fresh inline pulse with a `const` waveform. Configure its pulse settings directly in the tab.

## Calibration updates

Analysis and writeback use the fixed parameter source of the selected Run or Load result. Run keeps the source used for that measurement.

For a file from the same opened entry, Load reads the current persisted values at the file's historical point, using its role assignments. It does not use the tab's Open/Reset aliases or restore an unsaved historical parameter version. Files from another entry or without a point have no parameter source for analysis or writeback. Analysis that needs no such source remains available. Existing proposals stay attached to their captured destination when point or focus changes.

After the run, drag two lines on the 2D map to mark the half-flux and integer-flux sweet spots, then click Done. This is an interactive pick based on operator or agent judgement of the measured map. The preview proposes `qubit.flux_half`, `qubit.flux_int`, and `qubit.flux_period` = 2 × |`qubit.flux_int` − `qubit.flux_half`| for the source qubit when that source is available. Review the figure before applying these updates.

## MCP recipe

`onetone_spectrum_over_flux` requires explicit, non-null `flux_device`, `flux_unit`, `flux_range`, and `flux_points`. The range contains two distinct device-native endpoints, and `flux_points` must be at least 2. The unit must exactly match the cached device snapshot, including an explicit `native` when that is the device unit. Supply coordinates in that unit and connect the device before running the recipe.

The recipe can omit `center_mhz` and `span_mhz`. Omitted center uses current `$resonator.freq`; omitted span uses 2 times current `$resonator.kappa` for the full frequency span. The recipe reads these values during preparation and sets fixed numeric endpoints at center ± span/2. Omitted `freq_points` retains the GUI count.

## Recommended use

Run after lookback and an initial onetone/freq, before any twotone flux mapping, to find `$qubit.flux_int` and the period from the resonator map.

Review `flux_dev`. Opening a tab or pressing Reset resolves `$flux_dev` once from the alias table to a connected-device name. A missing or unbound alias leaves the required field empty and Invalid. Confirm the device and `flux_unit`, which starts as `A`. The selected unit labels the values as entered. Supply device-native values in that unit.

The calibrated sweep uses 101 flux points across the padded interval between `$qubit.flux_int` and `$qubit.flux_half`, and 101 frequency points at `$resonator.freq` ± `$resonator.kappa` MHz. Start at a low readout gain, about 0.005, to stay below punch-out so the dip tracks cleanly. If poor SNR makes the dip hard to see, raise the gain toward 0.05 while keeping the frequency window tight.

Survey wide, then narrow around a sweet spot. If the resonator's flux dispersion, or dispersive shift, looks small, narrow the frequency window early. For consecutive flux sweeps, reverse the sweep direction by swapping start and stop on the next run. This reduces the current source's return ramp across the full range.

Inspect the 2D map with `gui_tab_get_figure` and `subtab_id='run'` to judge the frequency window, SNR, and shift. After accepting `$qubit.flux_int`, move the flux device there and rerun onetone/freq at that flux before twotone/freq.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Supply resonator frequency/linewidth or explicit center_mhz/span_mhz. Confirm the cached flux device unit and native range.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, Reset to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, Reset and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `onetone_spectrum_over_flux` after inspecting the same Form errors.
   flux_device, flux_unit, flux_range and flux_points are required. freq_points and flux_points map to Expts. The interactive picker uses the existing GUI parameters.
   Common library/inline inputs select sources; overrides apply last. Null optional inputs retain Form values.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
