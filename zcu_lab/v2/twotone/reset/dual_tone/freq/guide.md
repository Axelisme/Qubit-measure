## Behavior

**Readers:** operator, agent. Dual-tone reset frequency map: a 2D sweep of the two reset-tone frequencies (pulse1 × pulse2 of the tested two-pulse reset), imaging the qubit response to locate the pair of sideband frequencies that jointly dump the excitation. Runs on real hardware; the hardware sweeps both frequency axes.

**Last updated:** 2026-10-21

## Parameter sources

Seeds two 201-point axes at `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2`, each +/- 50 MHz. Tested gains read `$qubit.ext.reset_gain1` and `$qubit.ext.reset_gain2`. Relax delay starts at 5*T1. Open and Reset fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the two_pulse_reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. Readout uses the captured `$qubit.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the fitted frequencies to `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2` only. The dual-tone power experiment proposes the tested reset module after analysing gains; choose reset length manually from the length-response plot.

## Recommended

Keep each axis a tight span (a few MHz) around the predicted sideband; the 2D scan cost grows with both axes. Analysis denoises the map before peak finding; wavelet smoothing is the default and Gaussian remains available for comparison.
