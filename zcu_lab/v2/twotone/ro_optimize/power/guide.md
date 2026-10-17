## Behavior

Readout power optimization: with the qubit toggled between g and e by a pi pulse, sweeps the readout gain and measures the g/e signal-to-noise ratio (SNR), to pick the readout power that best distinguishes the states. Runs on real hardware. A readout-tuning step, typically after the readout frequency is set.

## Expects (parameter context)

Seeds 101 gain points from 0.001 to 0.2. Relax delay reads 5.0 * `$qubit.t1.` Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

The qubit pulse uses `$qubit.module.x180` as a captured library link, otherwise a fresh pi_pulse template with a const waveform. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline readout template with a const waveform. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the SNR-maximizing readout gain into parameter context `$qubit.ext.best_ro_gain` (a.u.). When a cfg snapshot with pulse readout is available and `$qubit.ext.best_ro_freq` / `$qubit.ext.best_ro_gain` / `$qubit.ext.best_ro_length` are known from this result plus parameter context, also proposes `$qubit.readout.readout_dpm.`

## Recommended

Analysis denoises the SNR curve before picking the peak; wavelet smoothing is the default. The 'power penalty ratio' down-weights high gains (SNR × exp(-gain × ratio)), biasing the choice toward lower power to limit measurement-induced effects; ~0.5 is a sensible default.
