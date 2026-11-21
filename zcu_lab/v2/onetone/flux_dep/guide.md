**Readers:** operator, agent. Measure one-tone resonator flux dependence and locate flux sweet spots.

# One-tone flux dependence

Run sets the flux-bias device and sweeps the readout RF frequency. At each frequency, a shot plays the readout pulse and integrates its demodulated I and Q over the ADC window. Run averages the complex IQ before stepping to the next bias. Use the resonator's dispersive shift with bias to locate a fluxonium's half- and integer-flux symmetry points.

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## Parameter sources

Opening a tab or choosing Defaults takes a fixed copy of the current parameters and evaluates starting expressions once into editable input values. The frequency sweep starts with 101 points at `$resonator.freq` ± `$resonator.kappa`, in MHz. The readout uses `$resonator.freq`, `$resonator.wiring.ch`, and `$resonator.wiring.ro_ch` for its frequency and channels. The ADC window starts at 0.97 µs. Trigger timing reads the top-level `$resonator.time_of_fly`, adding `5/(2π·kappa)` µs when kappa is set. Missing or null TOF leaves trigger and pulse length required and blank. Pulse length starts at the larger of 1.0 µs and trigger plus the ADC window.

The flux sweep starts with 101 points between the selected device's actual integer and half, padded by 10% on each side. The lab helper projects `$qubit.flux: {offset, period}` onto the selected device with other devices' detached Form values. Missing other values contribute zero; missing offset or selected period leaves required endpoints blank. Missing or invalid required sources leave the affected fields empty and Invalid, blocking the run until valid values are supplied. For first bring-up before calibration, enter safe, user-approved frequency and flux bounds.

The sweet-spot picker projects `$qubit.flux` from the selected Run or Load capture onto the acquired cfg's device, using other devices' Run-start snapshot values. Missing values contribute zero; invalid values raise an error. Missing selected calibration uses the picker's default lines. Choose sweet spots from the measured map.

## Readout module

The Readout template starts inline with a constant envelope and gain 0.05 a.u. Gain is a DAC full-scale fraction. The sweep overrides the module frequency at every bias. Use the trigger offset and pulse duration to place the ADC window inside the arriving pulse. The ADC window is 0.03 µs shorter than the default 1.0 µs template pulse. Set Post-delay to let the cavity empty after readout; start at `5/(2π·κ)` µs for linewidth κ in MHz.

**Flux-bias device** chooses the source. Its **Bias unit** is A, V or the source's native units. **Flux bias** Start and Stop use that unit, not flux quanta. **Readout RF frequency** Start and Stop use absolute MHz, not mixer offsets. Expts is the number of values on either axis. Reps starts at 1000 shots per frequency and bias per round. Rounds starts at 10 frequency sweeps at each bias, and Relax delay starts at 1 µs after each shot.

## Read the maps

The Run heatmap colors the magnitude of the mean complex IQ, `|mean(I + iQ)|` in a.u., with flux bias on the horizontal axis and acquired readout RF frequency (MHz) on the vertical axis. Blank columns have no measurements. The title counts columns with data, including the current one. The trace below shows the latest bias's frequency cut. The magnitude is not a population. Neither the heatmap nor the frequency cut uses shot SEM; look for where the spectral notch moves.

Use the picker to mark the symmetry points of the notch track. Its interactive frequency display uses GHz; multiply by 1000 to compare it with the Run RF axis. The red line marks half flux and the blue line marks integer flux. The final Analyze map returns to MHz and the Run color scale. It labels the accepted lines and reports **Bias period** as twice their separation, in Bias unit. For example, if half flux is at 0.0010 A and integer flux at −0.0030 A, the period magnitude is 0.0080 A. The picks are not a fit and carry no fitted standard error. Recenter or extend the bias sweep if it lacks both symmetry points or if a pick lands at an edge. If the notch is clipped in frequency, extend the affected RF endpoint before judging symmetry.

Analyze identifies the source run and lists its configured Reps × Rounds. A loaded run with no compatible cfg has no averaging line; the plot labels the horizontal axis in device units rather than guessing amperes or volts.

## Calibration updates

Analysis and writeback use the fixed parameter source of the selected Run or Load result. Run keeps the source used for that measurement.

For a file from the same opened entry, Load reads the current persisted values at the file's historical point, using its role assignments. It does not use the tab's Open/Defaults aliases or restore an unsaved historical parameter version. Files from another entry or without a point have no parameter source for analysis or writeback. Analysis that needs no such source remains available. Existing proposals stay attached to their captured destination when point or focus changes.

After the run, drag two lines on the 2D map to mark the half-flux and integer-flux sweet spots, then click Done. This is an interactive pick based on operator or agent judgement of the measured map. With Run capture available, the preview separately proposes `flux.period.<cfg.flux_dev>` and a default-selected `flux.offset`. Period updates its magnitude and preserves its sign, initially positive.

Offset uses `0.5 - half_fit / period_s - Σ V_o / period_o`. Other values come from the Run-start snapshot. The swept value is replaced by half_fit once. The note lists other present calibrated devices' values and contributions, including zero values. Missing device values contribute zero, and other periods stay unchanged. An ordinary record stores the raw picked half/integer/positive period, scanned device, complete Run snapshot and original `{offset, period}` frame. Missing the whole snapshot blocks calibration. Review period, offset and record items separately; period can be accepted without offset. Acceptance does not read live devices or current parameters.

## MCP recipe

`onetone_spectrum_over_flux` requires explicit, non-null `flux_device`, `flux_unit`, `flux_range`, and `flux_points`. The range contains two distinct device-native endpoints, and `flux_points` must be at least 2. The unit must exactly match the cached device snapshot, including an explicit `native` when that is the device unit. Supply coordinates in that unit and connect the device before running the recipe.

Supply `center_mhz` and `span_mhz` together to set numeric endpoints at center ± span/2. Omitting both retains the Form's captured sweep. Supplying only one returns `needs_parameters` naming the other; the recipe does not infer it from calibration. Explicit null clears the endpoints. Omitting the sample count retains the Form count.

## Recommended use

Run after lookback and an initial onetone/freq, before any twotone flux mapping, to find the selected device's integer reference and period from the resonator map.

Opening the tab or choosing Defaults takes the initial Flux-bias device from the qubit's saved binding. Confirm that source and Bias unit, which starts at A. If a source is missing, fill the blank field before running. For a first sweep, start below the drive strength where the resonance shifts toward the bare center. For example, try gain 0.005 a.u. and increase toward 0.05 a.u. only if the notch is too faint and the device calibration allows it.

Survey wide, then narrow around a sweet spot. If the resonator's flux dispersion, or dispersive shift, looks small, narrow the frequency window early. For consecutive flux sweeps, reverse the sweep direction by swapping start and stop on the next run. This reduces the current source's return ramp across the full range.

Inspect the Run map before accepting a pick. After accepting the calibration, recompute the integer bias for the other devices' actual values. Move the scanned device to that bias and rerun onetone/freq there before twotone/freq.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Supply resonator frequency/linewidth or explicit center_mhz/span_mhz. Confirm the cached flux device unit and native range.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, choose Defaults to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, choose Defaults and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `onetone_spectrum_over_flux` after inspecting the same Form errors.
   flux_device, flux_unit, flux_range and flux_points are required. freq_points and flux_points map to Expts. The interactive picker uses the existing GUI parameters.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Defaults may replace unsaved inputs.
