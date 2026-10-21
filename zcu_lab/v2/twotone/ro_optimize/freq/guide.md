## Behavior

**Readers:** operator, agent. Readout frequency optimization: with the qubit prepared in g and e (a pi pulse toggles it), sweeps the readout frequency and measures the g/e signal-to-noise ratio (SNR), so you can pick the readout frequency that best distinguishes the two states. Runs on real hardware. One step of readout tuning; usually run after the qubit and a pi pulse are calibrated.

**Last updated:** 2026-10-21

## Parameter sources

Seeds 301 frequency points at `$resonator.freq` +/- `$resonator.kappa`. Relax delay reads 5.0 * `$qubit.t1`. Open and Reset fill editable starting values from the selected aliases and calibration. Qubit pulse templates read `$qubit.freq` and `$qubit.wiring.ch`. Readout templates read `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

The qubit pulse uses the `$qubit.module.x180` slot captured at Open/Reset, or a fresh pi_pulse template with a constant waveform. Readout uses the captured `$qubit.module.readout` slot, or a fresh readout template with a constant waveform. Its required parameter sources must still be present. Optional reset uses `$qubit.module.reset`. An unset slot, missing module or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the SNR-maximizing readout frequency to `$qubit.ext.best_ro_freq` (MHz). When acquired run settings include pulse readout and `$qubit.ext.best_ro_freq` / `$qubit.ext.best_ro_gain` / `$qubit.ext.best_ro_length` are known from this result plus captured calibration, also proposes `$qubit.readout.readout_dpm`.

## Recommended

Analysis denoises the SNR curve before picking the peak. Wavelet smoothing is the default; Gaussian smoothing provides a sigma-based comparison. A 'smooth' strength around 2 tames noise without washing out the feature.
