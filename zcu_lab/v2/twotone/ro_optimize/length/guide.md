## Behavior

Readout length optimization: with the qubit toggled between g and e by a pi pulse, sweeps the readout window length and measures the g/e signal-to-noise ratio (SNR), to pick the shortest readout that still resolves the states well. Runs on real hardware. A readout-tuning step, typically after the readout frequency and power are set.

## Expects (parameter context)

Reads from the parameter context (all optional): 'r_f' / 'best_ro_freq' — resonator / chosen readout frequency (~4000–8000 MHz); 'res_ch' / 'ro_ch' — drive / ADC channels; 'timeFly' — trigger-offset cable delay; 'q_f' / 'qub_ch' — qubit frequency / channel for the g↔e pi pulse.

## Expects (ModuleLibrary)

Needs a qubit-probe pulse module (typically a calibrated pi pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. 'readout_rf', usually pinned to the chosen readout frequency and gain); references a ModuleLibrary waveform 'ro_waveform' when present. Optionally references a reset module.

## Typical writeback

Proposes the SNR-maximizing readout length into parameter context 'best_ro_length' (us). When a cfg snapshot with pulse readout is available and 'best_ro_freq' / 'best_ro_gain' / 'best_ro_length' are known from this result plus parameter context, also proposes ModuleLibrary 'readout_dpm'.

## Recommended

Analysis denoises the SNR curve before picking the peak; wavelet smoothing is the default. The optional 'duration_t0' analyze param is the fixed-duration term in SNR/sqrt(length + t0): leave it blank (None) for the raw max; a small positive us value applies stronger short-readout bias than a large positive value.
