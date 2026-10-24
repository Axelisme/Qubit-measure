# MIST probe frequency and gain

**Readers:** operator, agent. Measure the MIST response over probe frequency and gain from the GUI or MCP.

**Last updated:** 2026-10-21

## Measurement

Open `mist/power_dep/drive_freq` to map the readout response after a probe pulse.
Each repetition performs reset, the optional initialization pulse, the probe,
then readout. Both probe frequency and gain sweep on the board.
The saved result is mean complex IQ shaped [frequency, gain]. Use a
classified-population experiment for calibrated populations or leakage
probabilities.

## Sources and defaults

Open and Reset capture the current aliases, parameter sources and module
selections. Reset discards edits and captures sources again. Run uses the
displayed configuration. Open or Reset again to apply alias changes.

- Optional reset and initialization use `$qubit.module.reset` and
  `$qubit.module.x180`. Missing slots leave these modules disabled.
- Probe starts from the resonator probe template: resonator frequency and
  generator channel, gain 0.05, and a 1 us constant waveform.
- Readout uses `$qubit.module.readout` when available, otherwise the readout
  template. The template needs resonator frequency, generator/ADC channels and
  `$resonator.time_of_fly`. Missing required sources stay Invalid.
- Frequency starts at `$resonator.freq` ± 1.5 times
  `$resonator.kappa`, with 51 points. Gain starts at 0–1 with 51 points.
  Both sweep axes override the probe's scalar frequency and gain during Run.
- Reps starts at 1000 and Rounds at 100. Relax delay starts at five times
  `$qubit.t1`. Missing/null T1 uses a 30 us scale, giving 150 us.
  Invalid stored T1 leaves the delay empty. All these values remain editable.

Review probe duration, channels, gain range, readout and waiting time before Run.
Set safe ranges using the limits of your hardware. Choose the frequency
range for the line of interest using measured resonances.

## Analysis and saving

Analyze subtracts one complex reference pooled over every frequency and the
first 5% of gain points (at least one), then plots the magnitude as `fit`.
Gain ordering therefore determines the reference region. The result is a
response-magnitude map. Calibration parameters remain unchanged. Use a
classified-population experiment for population calibration.

Save uses the native data format. Frequencies are MHz in memory and Hz on disk.
Load and Analyze use the saved record while the current form keeps its settings.
Analysis also accepts records with missing historical configuration.

The `mist_drive_frequency` recipe takes explicit two-endpoint `freq_range`
(MHz) and `gain_range` (normalized amplitude), plus optional `target` and
`points`. It uses captured GUI module selections and averaging defaults, runs
once, saves raw IQ, analyzes and captures a preview. Cancellation stops before
save and analysis.

## Offline checks

A white-noise simulation exercises pulse execution, acquisition, plots and
saving with a simulated device. Its output is noise only. Establish physical
MIST, heating or leakage thresholds with a suitable physical model or hardware
measurement.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Enter the frequency/gain rectangle and verify probe/readout timing. Supply GE calibration where the Form requests it.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, Reset to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, Reset and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `mist_drive_frequency` after inspecting the same Form errors.
   freq_range and gain_range are required pairs. points maps to Expts on both axes. Analysis has no operator parameters.
   The Form chooses module sources. Omitted inputs retain Form values; null clears/disables fields. Overrides apply last, then check_ready reports missing or invalid fields before Run.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
