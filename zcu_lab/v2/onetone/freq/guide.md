## Behavior

One-tone resonator spectroscopy: sweeps the readout frequency and fits the resonator response to extract its frequency and linewidth. Runs on real hardware. Typically run after a coarse resonator search has placed you near the right frequency.

## Expects (parameter context)

Uses the tab's captured resonator parameters: freq and kappa seed the sweep; ext.default_probe_length and ext.time_of_fly seed ADC length minus 0.1 us and trigger offset plus 0.05 us; wiring.ch and wiring.ro_ch seed channels. Missing or null calibration leaves required fields Invalid, without numeric fallback. The Sampling mode selector offers Linear or Homophasal. Homophasal shows three required fields: r_f in MHz, rf_w in MHz, and theta0 in rad. Fresh branches seed direct values from captured resonator.freq, resonator.kappa, and resonator.ext.theta0. Missing or unevaluable sources leave the corresponding fields empty. Evaluable values that fail core validation remain visible with an error and block submission. You can override each value directly or with an expression. Analysis optionally reads the route-matched resonator.ext.edelay prior from the same fixed capture.

## Expects (ModuleLibrary)

Starts with a fresh inline pulse readout and a const waveform. It does not adopt a calibrated library module or named waveform.

## Typical writeback

Proposes fitted values to resonator.freq, resonator.kappa, and resonator.ext.theta0 through the captured role binding. With a pulse-readout run snapshot, it also proposes resonator.readout.readout_rf from that template, changing only pulse/readout frequency while preserving gain, lengths, channels, and trigger timing. The compound resonator.ext.edelay proposal keeps delay and route together; it requires an identifiable absolute branch or a trusted prior/manual seed.

## MCP recipe

`onetone_spectrum` can omit `center_mhz` and `span_mhz`. Omitted center uses
current `$resonator.freq`; omitted span uses 3 times current `$resonator.kappa`
for the full frequency span. The recipe reads these values during preparation
and writes numeric endpoints at center ± span/2, not live formulas. Omitted
`points` retains the GUI count.

## Recommended

Analysis defaults to the hanger-model fit ('hm') with multiplicative amplitude-background fitting on. A sweep spanning a few linewidths around the known resonator frequency usually captures the dip cleanly; widen it if the resonator has drifted. Use homophasal sampling after a fit has written 'theta0' when you want equal resonator-circle phase spacing instead of a linear frequency grid. Select Homophasal to view the captured calibration values and fill any missing fields before Run. There is no separate Enabled switch. Selecting Linear removes the calibration fields and outputs no homophasal parameters. Switching back to Homophasal uses fresh defaults from the same capture, not your previous edits. Source changes affect a new Open or Reset, not an existing Form.

## Analysis controls

Analysis has four controls: Model type, Fit amplitude background, Fit phase curvature, and Electrical delay seed in us. Their defaults are hm, on, off, and blank.

Leave the delay blank to use a finite route-matched calibration as the branch seed. When no matching prior exists, analysis searches with an internal 100 us cap. Enter a finite delay to seed the fitted branch instead. Zero and negative seeds are valid. A seed is not a fixed delay. Nonfinite and nonnumeric delay inputs are invalid.
