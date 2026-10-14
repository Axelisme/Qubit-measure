## Behavior

Readout frequency optimization: with the qubit prepared in g and e (a pi pulse toggles it), sweeps the readout frequency and measures the g/e signal-to-noise ratio (SNR), so you can pick the readout frequency that best distinguishes the two states. Runs on real hardware. One step of readout tuning; usually run after the qubit and a pi pulse are calibrated.

## Expects (parameter context)

Reads from the parameter context (all optional): 'r_f' — resonator frequency, the sweep centre (~4000–8000 MHz); 'rf_w' — linewidth, setting the span as r_f ± 1.5*rf_w (~5–50 MHz; falls back to ±30 MHz when absent); 'res_ch' / 'ro_ch' — drive / ADC channels; 'timeFly' — cable time-of-flight for the trigger offset; 'q_f' / 'qub_ch' — qubit frequency / drive channel for the g↔e pi pulse.

## Expects (ModuleLibrary)

Needs a qubit-probe pulse module (typically a calibrated pi pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. 'readout_rf'); references a ModuleLibrary waveform named 'ro_waveform' when present. Optionally references a reset module.

## Typical writeback

Proposes the SNR-maximizing readout frequency into parameter context 'best_ro_freq' (MHz). When a cfg snapshot with pulse readout is available and 'best_ro_freq' / 'best_ro_gain' / 'best_ro_length' are known from this result plus parameter context, also proposes ModuleLibrary 'readout_dpm'.

## Recommended

Analysis denoises the SNR curve before picking the peak. Wavelet smoothing is the default; switch to Gaussian only when you need to compare against the older sigma-based result. A 'smooth' strength around 2 tames noise without washing out the feature.
