## Behavior

Readout frequency–power joint optimization: with the qubit toggled between g and e by a pi pulse, runs a 2D sweep of readout frequency × readout gain and measures the g/e signal-to-noise ratio (SNR), picking the (freq, gain) pair that best resolves the states. Runs on real hardware. Use this to refine freq and power together once you have a rough readout frequency.

## Expects (parameter context)

Reads from the parameter context (all optional): 'r_f' / 'best_ro_freq' — resonator / chosen readout frequency centring the freq sweep (~4000–8000 MHz); 'rf_w' — linewidth, setting the freq half-span (~5–50 MHz); 'res_ch' / 'ro_ch' — drive / ADC channels; 'timeFly' — trigger-offset cable delay; 'q_f' / 'qub_ch' — qubit frequency / channel for the g↔e pi pulse.

## Expects (ModuleLibrary)

Needs a qubit-probe pulse module (typically a calibrated pi pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. 'readout_rf'); references a ModuleLibrary waveform 'ro_waveform' when present. Optionally references a reset module.

## Typical writeback

Proposes the SNR-maximizing readout frequency and gain into parameter context 'best_ro_freq' (MHz) and 'best_ro_gain' (a.u.). When a cfg snapshot with pulse readout is available and 'best_ro_freq' / 'best_ro_gain' / 'best_ro_length' are known from this result plus parameter context, also proposes ModuleLibrary 'readout_dpm'.

## Recommended

Analysis denoises the 2D SNR map before picking the peak. Wavelet smoothing is the default; switch to Gaussian only when comparing against the older sigma-based result. Keep the freq span tight and the gain range modest so the 2D scan stays affordable.
