## Behavior

**Readers:** operator, agent. Dual-tone reset gain map: a 2D sweep of the two reset-tone gains (pulse1 × pulse2 of the tested two-pulse reset) at the calibrated sideband frequencies, imaging the residual excitation to pick the gain pair that resets most completely. Runs on real hardware.

**Last updated:** 2026-10-21

## Parameter sources

Tested frequencies read `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2`. Seeds two 51-point gain axes from 0.0 to 1.0, with relax delay starting at 5*T1. Open and Reset fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the two_pulse_reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. Readout uses the captured `$qubit.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the best gains to `$qubit.ext.reset_gain1` and `$qubit.ext.reset_gain2`. With acquired run settings and captured calibration, proposes `$qubit.reset.reset_120` from the tested reset, replacing only the two gains with this analysis. Tested frequencies and all other fields remain. An optional `$qubit.module.reset` row selects it.

## Recommended

Sweep each gain across its full usable range. Analysis denoises the map before picking the optimum; wavelet smoothing is the default and Gaussian remains available for comparison. The frequencies are held at '`$qubit.ext.reset_f1`' / '`$qubit.ext.reset_f2`', so calibrate those first.
