## Behavior

**Readers:** operator, agent. Bath-reset cavity frequency/gain map: a 2D sweep of the bath reset's cavity-tone frequency × gain, with a 4-point pi/2 tomography phase cycle at each frequency/gain pair, imaging the residual coherence to locate the (freq, gain) pair that best dumps the excitation into the cavity bath. Runs on real hardware.

**Last updated:** 2026-10-21

## Parameter sources

Seeds 51 frequency points from `$resonator.freq` - 1.2 * `$qubit.ext.rabi_freq` to `$resonator.freq` - 0.8 * `$qubit.ext.rabi_freq`, and 51 gain points from 0.4 to 1.0. Relax delay reads 5.0 * `$qubit.t1`. The pi/2 phase offset starts at the recommended 90 degrees and remains editable. Run adds its four-point tomography phase cycle to the configured offset. Open and Reset fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Tested reset starts as an editable copy of the bath_reset template. Optional initialization uses `$qubit.module.x180`. An unset slot, missing module or unbound alias leaves it disabled and Valid. The pi/2 tone requires `$qubit.module.x90`; missing calibration leaves it Invalid. An editable copy starts with a 90-degree tomography offset. The saved module stays unchanged. Readout uses the captured `$qubit.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the best cavity gain and frequency to `$qubit.ext.reset_bath_gain` and `$qubit.ext.reset_bath_freq` only. The bath phase experiment proposes the tested ground/excited reset modules; choose reset length manually from the length-response plot.

## Recommended

Analysis denoises the 2D map before picking the peak; wavelet smoothing is the default and Gaussian remains available for comparison. Save writes one 3D HDF5 file with frequency, gain and tomography phase axes.
