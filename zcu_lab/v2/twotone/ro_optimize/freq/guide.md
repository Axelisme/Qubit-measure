## Behavior

Readout frequency optimization: with the qubit prepared in g and e (a pi pulse toggles it), sweeps the readout frequency and measures the g/e signal-to-noise ratio (SNR), so you can pick the readout frequency that best distinguishes the two states. Runs on real hardware. One step of readout tuning; usually run after the qubit and a pi pulse are calibrated.

## Expects (parameter context)

Seeds 301 frequency points at `$resonator.freq` +/- `$resonator.kappa.` Relax delay reads 5.0 * `$qubit.t1.` Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

The qubit pulse uses `$qubit.module.x180` as a captured library link, otherwise a fresh pi_pulse template with a const waveform. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline readout template with a const waveform. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the SNR-maximizing readout frequency into parameter context `$qubit.ext.best_ro_freq` (MHz). When a cfg snapshot with pulse readout is available and `$qubit.ext.best_ro_freq` / `$qubit.ext.best_ro_gain` / `$qubit.ext.best_ro_length` are known from this result plus parameter context, also proposes `$qubit.readout.readout_dpm.`

## Recommended

Analysis denoises the SNR curve before picking the peak. Wavelet smoothing is the default; switch to Gaussian only when you need to compare against the older sigma-based result. A 'smooth' strength around 2 tames noise without washing out the feature.
