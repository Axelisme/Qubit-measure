## Behavior

AllXY: runs the 21 standard pairs of I, X90, Y90, X180 and Y180 gates before readout. A calibrated qubit gives a staircase of ground, equator and excited levels; amplitude and detuning errors bend it in distinct patterns. Y gates are the X pulses with phase +90 deg. Runs on real hardware. Run after amplitude Rabi and Ramsey have calibrated the pi and pi/2 pulses and the frequency.

## Expects (parameter context)

Reads 't1' (us) to seed relax_delay as 5*t1 (fallback ~30 us). Pulse modules pull 'q_f' (~2000–6000 MHz) and 'qub_ch'; readout pulls 'r_f', 'res_ch' / 'ro_ch' and 'timeFly'.

## Expects (ModuleLibrary)

Needs an X90 pulse module (prefers the calibrated library pi/2 pulse 'pi2_amp' / 'pi2_len'), an X180 pulse module (prefers 'pi_amp' / 'pi_len'), and a readout module (calibrated 'readout_dpm' / 'readout_rf' / 'readout' / 'res_readout', else a blank pulse-readout). Optional reset references a library reset when present, else stays disabled. The identity gate is the X90 pulse at zero gain.

## Typical writeback

No writeback. The summary reports 'amplitude_error', the relative drive amplitude error (+0.02 means 2% too much rotation), and 'detune_param', the dimensionless detuning parameter of the fit model. 'power_err' and 'detune_err' are the mean Bloch-z deviation (ground +1, excited -1) each error causes over the 21 pairs; they are not gate infidelities. 'residual_rms' is the fit residual in the same units; a residual comparable to those deviations means the model does not fit, for example because T1/T2 decay matters. Fix power errors with amplitude Rabi or zig-zag and detuning errors with Ramsey.

## Recommended

Leave 'Fit g/e levels' off to take the ground and excited levels from the data range; turn it on when the measured levels are noisy or do not reach both states. Use many reps and rounds: each point is a single gate pair.
