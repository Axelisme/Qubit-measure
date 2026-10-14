## Behavior

Zig-zag gain scan: runs the zig-zag repetition sequence at each gain of the repeated pulse, and picks the gain where the signal stays flattest across repetitions. Runs on real hardware.

## Expects (parameter context)

Reads 'pi_gain' to seed the gain sweep as 0.8–1.2*pi_gain (fallback 0.4–0.6); with repeat on X90_pulse, set the sweep around 'pi2_gain' instead. 't1' seeds relax_delay as 5*t1.

## Expects (ModuleLibrary)

Needs an X90 pulse module (prefers the calibrated library pi/2 pulse 'pi2_amp' / 'pi2_len'), an X180 pulse module (prefers 'pi_amp' / 'pi_len'), and a readout module (calibrated 'readout_dpm' / 'readout_rf' / 'readout' / 'res_readout', else a blank pulse-readout). Optional reset references a library reset when present, else stays disabled.

## Typical writeback

No writeback. The summary reports 'best_value', the best gain for the repeated pulse; update the pulse module yourself.

## Recommended

Analysis picks the sweep value whose signal changes least across repetitions (smoothed sum of step differences). Set 'Search from' / 'Search to' to ignore edges of the sweep. Keep n_times small (~6); the sweep multiplies the run time.
