# MIST probe frequency and gain

## Measurement

Open `mist/power_dep/drive_freq` to map the readout response after a probe pulse.
Each repetition performs reset, the optional initialization pulse, the probe,
then readout. Both probe frequency and gain sweep on the board.
The saved result is mean complex IQ shaped [frequency, gain], not classified
populations or a calibrated leakage probability.

## Sources and defaults

Open and Reset capture the current aliases, parameters and module library.
Reset discards edits and captures sources again. Run uses the displayed cfg;
changing an alias does not silently update an existing form.

- Optional reset and initialization use `$qubit.module.reset` and
  `$qubit.module.x180`. Missing slots leave these modules disabled.
- Probe starts from the resonator probe template: resonator frequency and
  generator channel, gain 0.05, and a 1 us constant waveform.
- Readout uses `$qubit.module.readout` when available, otherwise the readout
  template. The template needs resonator frequency, generator/ADC channels and
  `$resonator.ext.time_of_fly`. Missing required sources stay Invalid.
- Frequency starts at `$resonator.freq` ± 1.5 times
  `$resonator.kappa`, with 51 points. Gain starts at 0–1 with 51 points.
  Both sweep axes override the probe's scalar frequency and gain during Run.
- Reps starts at 1000 and Rounds at 100. Relax delay starts at five times
  `$qubit.t1`. Missing/null T1 uses a 30 us scale, giving 150 us.
  Invalid stored T1 leaves the delay empty. All these values remain editable.

Review probe duration, channels, gain range, readout and waiting time before Run.
These are starting values, not hardware safety limits. Choose the frequency
range for the line of interest; do not infer its location from simulator truth.

## Analysis and saving

Analyze subtracts one complex reference pooled over every frequency and the
first 5% of gain points (at least one), then plots the magnitude as `fit`.
Gain ordering therefore determines the reference region. The plot does not fit
a transition or produce a population calibration. There is no writeback.

Save uses the existing native persistence format. Frequencies are MHz in
memory and Hz on disk. Load and Analyze use the saved record, not the current
form; analysis also accepts a record without cfg.

The `mist_drive_frequency` recipe takes explicit two-endpoint `freq_range`
(MHz) and `gain_range` (normalized amplitude), plus optional `target` and
`points`. It uses captured GUI module/averaging defaults, runs once, saves raw IQ,
analyzes and captures a preview. Cancellation stops before save and analysis.

## Offline checks

A white-noise simulation can exercise compilation, acquisition, plots and
persistence without hardware. It does not model MIST physics, heating or
leakage, so a flat/noisy map is not evidence for a physical threshold.
