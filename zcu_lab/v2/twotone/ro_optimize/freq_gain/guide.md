## Behavior

Readout frequency–power joint optimization: with the qubit toggled between g and e by a pi pulse, runs a 2D sweep of readout frequency × readout gain and measures the g/e signal-to-noise ratio (SNR), picking the (freq, gain) pair that best resolves the states. Runs on real hardware. Use this to refine freq and power together once you have a rough readout frequency.

## Expects (parameter context)

Seeds 31 frequency points at `$resonator.freq` +/- 0.5 * `$resonator.kappa`, not `$qubit.ext.best_ro_freq.` Seeds 31 gain points from 0.0 to 0.2. Relax delay reads 5.0 * `$qubit.t1.` Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Missing or invalid required sources leave the affected fields empty and Invalid; supply calibration or direct values before Run. No numeric calibration fallback is supplied.

## Expects (ModuleLibrary)

The qubit pulse uses `$qubit.module.x180` as a captured library link, otherwise a fresh pi_pulse template with a const waveform. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline readout template with a const waveform. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the SNR-maximizing readout frequency and gain into parameter context `$qubit.ext.best_ro_freq` (MHz) and `$qubit.ext.best_ro_gain` (a.u.). When a cfg snapshot with pulse readout is available and `$qubit.ext.best_ro_freq` / `$qubit.ext.best_ro_gain` / `$qubit.ext.best_ro_length` are known from this result plus parameter context, also proposes `$qubit.readout.readout_dpm.`

## Recommended

Analysis denoises the 2D SNR map before picking the peak. Wavelet smoothing is the default; switch to Gaussian only when comparing against the older sigma-based result. Keep the freq span tight and the gain range modest so the 2D scan stays affordable.
