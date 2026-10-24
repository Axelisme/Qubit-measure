## Behavior

**Readers:** operator, agent. How to run one-tone resonator spectroscopy from the GUI or MCP and review calibration updates.

The measurement runs on real hardware. It sweeps readout frequency and fits the resonator response to extract its frequency and linewidth. Run it after a coarse resonator search has placed you near the right frequency.

**Last updated:** 2026-10-21

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Parameter sources

The tab uses a fixed copy of the resonator parameters taken when it was opened or reset. `resonator.freq` and `resonator.kappa` seed the sweep. The ADC window starts at 0.99 us. Trigger timing reads the top-level `$resonator.time_of_fly`, adding `5/(2π·kappa)` us when kappa is set. Missing or null TOF leaves trigger and pulse length required and blank. Pulse length starts at the larger of 1.0 us and trigger plus the ADC window. `resonator.wiring.ch` and `resonator.wiring.ro_ch` seed the channels. Missing or null calibration leaves required fields Invalid until valid values are supplied.

The Sampling mode selector offers Linear or Homophasal. Homophasal shows three required fields: `r_f` in MHz, `rf_w` in MHz, and `theta0` in rad. Selecting Homophasal seeds editable values from the saved `resonator.freq`, `resonator.kappa`, and `resonator.ext.theta0`. Missing sources or expression evaluation failures leave the corresponding fields empty. Values that fail measurement validation stay visible with an error and block submission. Override each value directly or with an expression.

Analysis can also use the saved `resonator.ext.edelay` calibration when it matches the readout route.

## Readout module

The readout starts as a fresh inline pulse with a `const` waveform. Configure its pulse settings directly in the tab.

## Calibration updates

Analysis and writeback use the fixed parameter source of the selected Run or Load result. Run keeps the source used for that measurement.

For a file from the same opened entry, Load reads the current persisted values at the file's historical point, using its role assignments. It does not use the tab's Open/Reset aliases or restore an unsaved historical parameter version. Files from another entry or without a point have no parameter source for analysis or writeback. Analysis that needs no such source remains available. Existing proposals stay attached to their captured destination when point or focus changes.

Analysis proposes fitted values for `resonator.freq`, `resonator.kappa`, and `resonator.ext.theta0`. The selected result's parameter source identifies the destination resonator when that source is available. When the run includes pulse-readout settings, analysis also proposes a `resonator.readout.readout_rf` library module based on those settings. It changes the pulse/readout frequency and preserves gain, lengths, channels, and trigger timing.

When the selected result has a parameter source, an update to the `$qubit.module.readout` slot follows the module proposal. The slot value is the resolved `$resonator.readout.readout_rf` path from that same source. Apply both the module and slot updates to select that library readout for downstream experiments. Scalar and module proposals are also available for runs made without saved parameters.

The `resonator.ext.edelay` proposal keeps delay and route together. It requires an identifiable absolute branch or a trusted prior calibration or manual seed.

## MCP recipe

Supply `center_mhz` and `span_mhz` together to set numeric endpoints at center ± span/2. Omitting both retains the Form's captured sweep. Supplying only one returns `needs_parameters` naming the other; the recipe does not infer it from calibration. Explicit null clears the endpoints. Omitting the sample count retains the Form count.

## Recommended use

Analysis defaults to the hanger-model fit, `hm`, with multiplicative amplitude-background fitting on. A sweep spanning a few linewidths around the known resonator frequency usually captures the dip cleanly. Widen it if the resonator has drifted.

Use Homophasal after a fit has written `theta0` when you want equal resonator-circle phase spacing. Select Homophasal to view the saved calibration values and fill any missing fields before Run. The Sampling mode selector controls this directly. Linear uses a linear frequency grid and removes the homophasal fields from the submitted settings. Switching back to Homophasal resets those fields to defaults from the saved parameters, replacing previous edits. To use changed source parameters, reopen the tab or press Reset.

## Analysis controls

Analysis has four controls: Model type, Fit amplitude background, Fit phase curvature, and Electrical delay seed in us. Their defaults are `hm`, on, off, and blank.

Leave the delay blank to use a finite route-matched calibration as the branch seed. When a matching prior is absent, analysis searches with a 100 us cap. Enter a finite delay to seed the fitted branch instead. Valid delay inputs are finite numbers, including zero and negative values. The fit can adjust the seeded delay.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Supply a resonator frequency or an explicit center, plus a positive linewidth or explicit span. Check readout timing.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, Reset to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, Reset and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `onetone_spectrum` after inspecting the same Form errors.
   points maps to frequency Expts. Supply center_mhz/span_mhz together or omit both to retain the Form sweep. Analysis controls are model_type, fit_bg_amp_slope, fit_bg_phase_curvature and edelay.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
