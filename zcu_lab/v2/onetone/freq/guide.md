## Behavior

**Readers:** operator, agent. How to run one-tone resonator spectroscopy from the GUI or MCP and review calibration updates.

The measurement runs on real hardware. It sweeps readout frequency and fits the resonator response to extract its frequency and linewidth. Run it after a coarse resonator search has placed you near the right frequency.

**Last updated:** 2026-10-21

## Parameter sources

The tab uses a fixed copy of the resonator parameters taken when it was opened or reset. `resonator.freq` and `resonator.kappa` seed the sweep. `resonator.ext.default_probe_length` minus 0.1 us seeds the ADC length, and `resonator.ext.time_of_fly` plus 0.05 us seeds the trigger offset. `resonator.wiring.ch` and `resonator.wiring.ro_ch` seed the channels. Missing or null calibration leaves required fields Invalid until valid values are supplied.

The Sampling mode selector offers Linear or Homophasal. Homophasal shows three required fields: `r_f` in MHz, `rf_w` in MHz, and `theta0` in rad. Selecting Homophasal seeds editable values from the saved `resonator.freq`, `resonator.kappa`, and `resonator.ext.theta0`. Missing sources or expression evaluation failures leave the corresponding fields empty. Values that fail measurement validation stay visible with an error and block submission. Override each value directly or with an expression.

Analysis can also use the saved `resonator.ext.edelay` calibration when it matches the readout route.

## Readout module

The readout starts as a fresh inline pulse with a `const` waveform. Configure its pulse settings directly in the tab.

## Calibration updates

Analysis proposes fitted values for `resonator.freq`, `resonator.kappa`, and `resonator.ext.theta0` for the resonator bound when the tab was opened or reset. When the run includes pulse-readout settings, analysis also proposes a `resonator.readout.readout_rf` library module based on those settings. It changes the pulse/readout frequency and preserves gain, lengths, channels, and trigger timing.

When saved parameters are available, an update to the `$qubit.module.readout` slot follows the module proposal. The slot value is the resolved `$resonator.readout.readout_rf` path from those same saved parameters. Apply both the module and slot updates to select that library readout for downstream experiments. Scalar and module proposals are also available for runs made without saved parameters.

The `resonator.ext.edelay` proposal keeps delay and route together. It requires an identifiable absolute branch or a trusted prior calibration or manual seed.

## MCP recipe

`onetone_spectrum` can omit `center_mhz` and `span_mhz`. Omitted center uses current `$resonator.freq`; omitted span uses 3 times current `$resonator.kappa` for the full frequency span. The recipe reads these values during preparation and sets fixed numeric endpoints at center ± span/2. Omitted `points` retains the GUI count.

## Recommended use

Analysis defaults to the hanger-model fit, `hm`, with multiplicative amplitude-background fitting on. A sweep spanning a few linewidths around the known resonator frequency usually captures the dip cleanly. Widen it if the resonator has drifted.

Use Homophasal after a fit has written `theta0` when you want equal resonator-circle phase spacing. Select Homophasal to view the saved calibration values and fill any missing fields before Run. The Sampling mode selector controls this directly. Linear uses a linear frequency grid and removes the homophasal fields from the submitted settings. Switching back to Homophasal resets those fields to defaults from the saved parameters, replacing previous edits. To use changed source parameters, reopen the tab or press Reset.

## Analysis controls

Analysis has four controls: Model type, Fit amplitude background, Fit phase curvature, and Electrical delay seed in us. Their defaults are `hm`, on, off, and blank.

Leave the delay blank to use a finite route-matched calibration as the branch seed. When a matching prior is absent, analysis searches with a 100 us cap. Enter a finite delay to seed the fitted branch instead. Valid delay inputs are finite numbers, including zero and negative values. The fit can adjust the seeded delay.
