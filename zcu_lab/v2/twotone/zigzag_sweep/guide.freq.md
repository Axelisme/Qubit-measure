## Behavior

Zig-zag frequency scan: runs the zig-zag repetition sequence at each drive frequency of the repeated pulse, and picks the frequency where the signal stays flattest across repetitions. Runs on real hardware. The simulator measures drive phase from the start of each shot, while hardware uses absolute time.

## Expects (parameter context)

Reads 'q_f' to seed the frequency sweep as q_f ± 2 MHz (fallback 4998–5002 MHz). 't1' seeds relax_delay as 5*t1.

## Expects (ModuleLibrary)

Needs an X90 pulse module (prefers the calibrated library pi/2 pulse 'pi2_amp' / 'pi2_len'), an X180 pulse module (prefers 'pi_amp' / 'pi_len'), and a readout module (calibrated 'readout_dpm' / 'readout_rf' / 'readout' / 'res_readout', else a blank pulse-readout). Optional reset references a library reset when present, else stays disabled.

## Typical writeback

No writeback. The summary reports 'best_value', the best drive frequency (MHz); update 'q_f' or the pulse module yourself.

## Recommended

Analysis picks the sweep value whose signal changes least across repetitions (smoothed sum of step differences). Set 'Search from' / 'Search to' to ignore edges of the sweep. Keep n_times small (~6); the sweep multiplies the run time.
