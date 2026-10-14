## Behavior

Automatic readout optimization: a Bayesian optimizer (skopt) searches readout frequency, gain and length jointly to maximize the g/e signal-to-noise ratio (SNR), with the qubit toggled between g and e by a pi pulse. Runs on real hardware. Use it to find a good readout in one shot instead of tuning freq / power / length one axis at a time.

## Expects (parameter context)

Reads from the parameter context (all optional): 'r_f' / 'best_ro_freq' — resonator / chosen readout frequency centring the freq search (~4000–8000 MHz); 'rf_w' — linewidth, scaling the freq search range (~5–50 MHz); 'res_ch' / 'ro_ch' — drive / ADC channels; 'timeFly' — trigger-offset cable delay; 'q_f' / 'qub_ch' — qubit frequency / channel for the g↔e pi pulse.

## Expects (ModuleLibrary)

Needs a qubit-probe pulse module (typically a calibrated pi pulse, e.g. 'pi_amp') and a pulse-readout module (e.g. 'readout_rf'); references a ModuleLibrary waveform 'ro_waveform' when present. Optionally references a reset module.

## Typical writeback

Proposes the optimizer's best readout frequency, gain and length into parameter context 'best_ro_freq' (MHz), 'best_ro_gain' (a.u.) and 'best_ro_length' (us). When a cfg snapshot with pulse readout is available, also proposes ModuleLibrary 'readout_dpm'.

## Recommended

The three sweep axes define the optimizer's SEARCH BOUNDS (min / max), not a grid — keep them reasonably tight (e.g. freq within a fraction of a linewidth, gain and length in a sensible band) so the search converges. 'Optimizer points' (~1000) sets how many evaluations to spend; raise it for wider bounds, lower it for a quick search. No analysis knobs — it reports the best evaluated point.
