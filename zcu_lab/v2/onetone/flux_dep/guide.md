## Behavior

**Readers:** operator, agent. How to measure one-tone resonator flux dependence from the GUI or MCP and locate flux sweet spots.

The 2D sweep varies an external flux-bias device value and readout frequency to trace how the resonator frequency moves with flux. Use the map to locate a Fluxonium's half-flux and integer-flux sweet spots. The measurement runs on real hardware and requires a SoC connection and a configured flux-bias device.

**Last updated:** 2026-10-21

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Parameter sources

Opening a tab or pressing Reset takes a fixed copy of the current parameters and evaluates starting expressions once into editable input values. The frequency sweep starts with 101 points at `$resonator.freq` ± `$resonator.kappa`, in MHz. The readout uses `$resonator.freq`, `$resonator.wiring.ch`, and `$resonator.wiring.ro_ch` for its frequency and channels. The ADC window starts at 0.99 us. Trigger timing reads the top-level `$resonator.time_of_fly`, adding `5/(2π·kappa)` us when kappa is set. Missing or null TOF leaves trigger and pulse length required and blank. Pulse length starts at the larger of 1.0 us and trigger plus the ADC window.

The flux sweep starts with 101 points between the selected device's actual integer and half, padded by 10% on each side. The lab helper reads `$qubit.flux.<device>` and shifts the references for other contributors' detached Form device values. Missing or invalid required sources leave the affected fields empty and Invalid, blocking the run until valid values are supplied. For first bring-up before calibration, enter safe, user-approved frequency and flux bounds.

The sweet-spot picker reads the acquired cfg's device entry in `$qubit.flux` from the selected Run or Load capture. It shifts the half/integer references using other contributors' Run-start snapshot values. Missing selected calibration uses the picker's default lines; invalid or missing used contributors raise an error. Choose sweet spots from the measured map.

## Readout module

The readout starts as a fresh inline pulse with a `const` waveform. Configure its pulse settings directly in the tab.

## Calibration updates

Analysis and writeback use the fixed parameter source of the selected Run or Load result. Run keeps the source used for that measurement.

For a file from the same opened entry, Load reads the current persisted values at the file's historical point, using its role assignments. It does not use the tab's Open/Reset aliases or restore an unsaved historical parameter version. Files from another entry or without a point have no parameter source for analysis or writeback. Analysis that needs no such source remains available. Existing proposals stay attached to their captured destination when point or focus changes.

After the run, drag two lines on the 2D map to mark the half-flux and integer-flux sweet spots, then click Done. This is an interactive pick based on operator or agent judgement of the measured map. The preview proposes the source qubit's `flux.<cfg.flux_dev>` entry when the Run capture is available. It converts the picked half/integer to the other contributors' integer references using their Run-start snapshot biases. The period uses the new magnitude and stored sign, initially positive. Other device entries stay unchanged. An ordinary record accompanies the point update with raw picked half/integer/positive period, scanned device, complete connected-device Run snapshot and the original Run parameter frame. Missing Run snapshot blocks calibration. Review the figure and accept both proposals. Acceptance does not read live devices or current parameters.

## MCP recipe

`onetone_spectrum_over_flux` requires explicit, non-null `flux_device`, `flux_unit`, `flux_range`, and `flux_points`. The range contains two distinct device-native endpoints, and `flux_points` must be at least 2. The unit must exactly match the cached device snapshot, including an explicit `native` when that is the device unit. Supply coordinates in that unit and connect the device before running the recipe.

Supply `center_mhz` and `span_mhz` together to set numeric endpoints at center ± span/2. Omitting both retains the Form's captured sweep. Supplying only one returns `needs_parameters` naming the other; the recipe does not infer it from calibration. Explicit null clears the endpoints. Omitting the sample count retains the Form count.

## Recommended use

Run after lookback and an initial onetone/freq, before any twotone flux mapping, to find the selected device's integer reference and period from the resonator map.

Review `flux_dev`. Opening a tab or pressing Reset reads `$qubit.flux_source` once into an editable direct device name. A missing source leaves the required field empty and Invalid. Confirm the device and `flux_unit`, which starts as `A`. The selected unit labels the values as entered. Supply device-native values in that unit.

The calibrated sweep uses 101 flux points across the padded interval between the selected device's actual integer and half, and 101 frequency points at `$resonator.freq` ± `$resonator.kappa` MHz. Start at a low readout gain, about 0.005, to stay below punch-out so the dip tracks cleanly. If poor SNR makes the dip hard to see, raise the gain toward 0.05 while keeping the frequency window tight.

Survey wide, then narrow around a sweet spot. If the resonator's flux dispersion, or dispersive shift, looks small, narrow the frequency window early. For consecutive flux sweeps, reverse the sweep direction by swapping start and stop on the next run. This reduces the current source's return ramp across the full range.

Inspect the 2D map with `gui_tab_get_figure` and `subtab_id='run'` to judge the frequency window, SNR, and shift. After accepting the calibration, recompute the actual integer for other devices' current values and move the scanned device there and rerun onetone/freq at that flux before twotone/freq.

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
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
