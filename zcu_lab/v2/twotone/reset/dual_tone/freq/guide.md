**Readers:** operator, agent. Find a carrier pair for two simultaneous reset tones. The sequence is optional preparation reset → optional preparation pulse → both fixed-gain reset tones → readout after the longer tone. The GUI sweeps both RF axes in hardware.

**Last updated:** 2026-11-22

## Parameter sources

Seeds two 201-point axes at `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2`, each +/- 50 MHz. Tested gains read `$qubit.ext.reset_gain1` and `$qubit.ext.reset_gain2`. Relax delay starts at 5*T1. Open and Defaults fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.015 µs for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Defaults capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the two_pulse_reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. Readout uses the captured `$resonator.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Acquisition and figures

Reps starts at 1000 per sweep point and branch in each acquisition round; Rounds starts at 10. Check the requested Reps × Rounds in Analyze before comparing runs.

Both figures show the magnitude of mean IQ minus the median IQ background. The GUI map is the same observable in Live and Analyze. In a host-stepped notebook run, the extra live trace shows the latest acquired tone 2 row. The Analyze star marks the RF pair picked after smoothing; it sits on the original unsmoothed map. Wavelet or Gaussian smoothing changes peak selection, not the plotted data. These are differences between mean IQ values, not classified populations. A 2D map does not show per-cell SEM; the live title gives the number of acquired cells.

Read the two frequencies in axis order and use them as the next tested carriers. For a peak on a boundary, shift only the corresponding sweep toward that edge and repeat. For an interior peak, tighten each axis around the star until it resolves over multiple cells. Do not claim a reset fidelity from this contrast. Layout reference: Ding, Fig. 5-8(b) (printed p. 151) uses a heatmap of population versus drive frequency and pulse plateau. Here the color is an IQ-derived contrast and the axes are separate controls of the tested reset.

In Analyze, Peak selection filter chooses wavelet or Gaussian smoothing. Filter strength multiplies the wavelet threshold in a.u.; for Gaussian it is the standard deviation in sample steps. The star can move when these options change, although the plotted raw map stays the same.

## Typical writeback

Proposes the selected RF frequencies to `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2` only. The dual-tone power experiment proposes the tested reset module after analysing gains; choose reset length manually from the length-response plot.

## Recommended

Keep each axis a tight span (a few MHz) around the predicted sideband; the 2D scan cost grows with both axes. Analysis denoises the map before peak finding; wavelet smoothing is the default and Gaussian remains available for comparison.
