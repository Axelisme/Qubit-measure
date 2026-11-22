**Readers:** operator, agent. Choose a gain pair for two simultaneous reset tones at calibrated carriers. The sequence is optional preparation reset → optional preparation pulse → both reset tones → readout after the longer tone.

**Last updated:** 2026-11-22

## Parameter sources

Tested frequencies read `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2`. Seeds two 51-point gain axes from 0.0 to 1.0, with relax delay starting at 5*T1. Open and Defaults fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.015 µs for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Defaults capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 µs, giving a 150 µs wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the two_pulse_reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. Readout uses the captured `$resonator.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Acquisition and figures

Reps starts at 1000 per sweep point and branch in each acquisition round; Rounds starts at 10. Check the requested Reps × Rounds in Analyze before comparing runs.

Both figures show the magnitude of mean IQ minus the IQ at the minimum-gain corner. Analyze smooths the complex map to pick a gain pair, then marks it with a star on the unsmoothed map. It selects an extremum of the smoothed contrast depending on the first corner relative to the map mean; in a reversed scan the first corner need not be the minimum-gain corner. These are differences between mean IQ values, not classified populations. A 2D map does not show per-cell SEM; the live title gives the number of acquired cells.

Read the marked tone 1 and tone 2 gains in axis order (a.u.). If a candidate lies at an edge of a permitted range, check that it persists on a nearby repeat; move or extend that axis only within safe calibrated bounds. Confirm state preparation separately before applying a module. Layout reference: Ding, Fig. 5-8(b) (printed p. 151) uses a heatmap of population versus drive frequency and pulse plateau. Here the color is an IQ-derived contrast and the axes are separate controls of the tested reset.

In Analyze, Peak selection filter chooses wavelet or Gaussian smoothing. Filter strength multiplies the wavelet threshold in a.u.; for Gaussian it is the standard deviation in sample steps. The star can move when these options change, although the plotted raw map stays the same.

## Typical writeback

Proposes the best gains to `$qubit.ext.reset_gain1` and `$qubit.ext.reset_gain2`. With acquired run settings and captured calibration, proposes `$qubit.reset.reset_120` from the tested reset, replacing only the two gains with this analysis. Tested frequencies and all other fields remain. An optional `$qubit.module.reset` row selects it.

The proposal uses only this Run's tested reset and captured parameters, not current aliases, ext values or an older library module. Missing saved cfg or parameter capture leaves only the scalar proposals. Source and Apply errors still propagate. Applying the library rows without the optional reset-slot row leaves the selected reset unchanged.

## Recommended

Sweep each gain across its full usable range. Analysis denoises the map before picking the optimum; wavelet smoothing is the default and Gaussian remains available for comparison. The frequencies are held at '`$qubit.ext.reset_f1`' / '`$qubit.ext.reset_f2`', so calibrate those first.
