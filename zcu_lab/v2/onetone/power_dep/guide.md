## Behavior

**Readers:** operator, agent. How to measure one-tone resonator power dependence from the GUI or MCP and read the punch-out map.

The 2D sweep varies readout gain and readout frequency to map how the resonator response shifts with drive strength. Use it to find the punch-out or high-power transition and choose a low-power readout gain. The measurement runs on real hardware and requires a SoC connection.

**Last updated:** 2026-10-21

## Parameter sources

The tab uses a fixed copy of the resonator parameters taken when it was opened or reset. `resonator.freq` and `resonator.kappa` seed the frequency sweep. `resonator.wiring.ch` and `resonator.wiring.ro_ch` seed the generator and ADC channels. The ADC window starts at 0.99 us. Trigger timing reads the top-level `$resonator.time_of_fly`, adding `5/(2π·kappa)` us when kappa is set. Missing or null TOF leaves trigger and pulse length required and blank. Pulse length starts at the larger of 1.0 us and trigger plus the ADC window. Missing or null calibration and unbound aliases leave required fields Invalid until valid values are supplied. Other source failures are reported as errors.

## Readout module

The readout starts as a fresh inline pulse with a `const` waveform. Configure its pulse settings directly in the tab.

## Results

The run produces a 2D map for visual inspection. Read the punch-out power and low-power frequency from the map, then update parameters in a separate calibration step. Analysis and calibration updates are manual for this measurement.

## MCP recipe

`onetone_spectrum_over_power` can omit `center_mhz` and `span_mhz`. Omitted center uses current `$resonator.freq`; omitted span uses 3 times current `$resonator.kappa` for the full frequency span. The recipe reads these values during preparation and sets fixed numeric endpoints at center ± span/2. Omitted `freq_points` retains the GUI count.

## Recommended use

A typical sweep uses gain from about 0.001 to 0.5 over about 101 points, ordered from low to high power. Sweep frequency around `resonator.freq` by a few linewidths over about 201 points. `earlystop_snr` starts disabled. Leave it blank (`None`) to keep early stopping disabled. Zero and negative thresholds are rejected by the Form. Set it positive to stop a column early once the signal-to-noise target is reached. Narrow the gain range once you have located the transition.
