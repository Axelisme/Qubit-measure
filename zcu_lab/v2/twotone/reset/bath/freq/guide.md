**Readers:** operator, agent. Find the cavity RF frequency and gain that maximize bath-reset tomography contrast. Each shot uses optional preparation reset and preparation pulse, then simultaneous cavity and qubit drives, followed by a tomography π/2 pulse after the cavity tone and readout.

**Last updated:** 2026-11-22

## Parameter sources

Seeds 51 frequency points from `$resonator.freq` - 1.2 * `$qubit.ext.rabi_freq` to `$resonator.freq` - 0.8 * `$qubit.ext.rabi_freq`, and 51 gain points from 0.4 to 1.0. Relax delay reads 5.0 * `$qubit.t1`. The pi/2 phase offset starts at the recommended 90 degrees and remains editable. Run adds its four-point tomography phase cycle to the configured offset. Open and Defaults fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.015 µs for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Defaults capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the bath_reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. The pi/2 tone requires `$qubit.module.x90`; missing calibration leaves it Invalid. An editable copy starts with a 90-degree tomography offset. The saved module stays unchanged. Readout uses the captured `$resonator.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Acquisition and figures

Reps starts at 1000 per sweep point and branch in each acquisition round; Rounds starts at 10. Check the requested Reps × Rounds in Analyze before comparing runs.

At each cell, the tomography pulse uses 0°, 90°, 180° and 270° relative to the phase in Tested reset. The color value is |S180−S0|² + |S270−S90|², where S is the mean complex IQ at the indicated phase offset. Its unit is a.u.². This contrast detects phase-dependent response, not reset probability. The live title counts acquired cells; a 2D map has no per-cell shot SEM.

Analyze uses the same raw contrast and marks the maximum of the smoothed contrast with a star. Use its cavity gain and RF as the fixed settings for the next duration and phase sweeps. For an edge maximum, move the corresponding scan window toward the star within the permitted settings and recheck for an interior peak. Layout reference: Ding, Fig. 5-8(b) (printed p. 151) uses a heatmap of population versus drive frequency and pulse plateau. Here the color is an IQ-derived contrast and the axes are separate controls of the tested reset.

In Analyze, Peak selection filter chooses wavelet or Gaussian smoothing. Filter strength multiplies the wavelet threshold in a.u.; for Gaussian it is the standard deviation in sample steps. The star can move when these options change, although the plotted raw map stays the same.

## Typical writeback

Proposes the best cavity gain and frequency to `$qubit.ext.reset_bath_gain` and `$qubit.ext.reset_bath_freq` only. The bath phase experiment proposes the tested ground/excited reset modules; choose reset length manually from the length-response plot.

## Recommended

Analysis denoises the 2D map before picking the peak; wavelet smoothing is the default and Gaussian remains available for comparison. Save writes one 3D HDF5 file with frequency, gain and tomography phase axes.
