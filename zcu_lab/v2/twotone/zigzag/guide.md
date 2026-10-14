## Behavior

Zig-zag: applies an X90 pulse, then repeats the chosen pulse 0..n_times times before readout, amplifying small pulse-amplitude errors into a growing zig-zag of the signal versus repetition count. Runs on real hardware. Run after amplitude Rabi has calibrated the pi and pi/2 pulses.

## Expects (parameter context)

Reads 't1' (us) to seed relax_delay as 5*t1 (fallback ~30 us). Pulse modules pull 'q_f' (~2000–6000 MHz) and 'qub_ch'; readout pulls 'r_f', 'res_ch' / 'ro_ch' and 'timeFly'.

## Expects (ModuleLibrary)

Needs an X90 pulse module (prefers the calibrated library pi/2 pulse 'pi2_amp' / 'pi2_len'), an X180 pulse module (prefers 'pi_amp' / 'pi_len'), and a readout module (calibrated 'readout_dpm' / 'readout_rf' / 'readout' / 'res_readout', else a blank pulse-readout). Optional reset references a library reset when present, else stays disabled.

## Typical writeback

No analysis and no writeback. Preparation coherence, contrast, and drive history can also change the trace; compare controls before changing gate calibration.

## Recommended

Repeat on X180_pulse to check the pi pulse; repeat on X90_pulse (applied in pairs) to check the pi/2 pulse. About 10 repetitions usually show a mis-calibration clearly. Use the zig-zag scan adapters to sweep the repeated pulse's gain or frequency. reset_phase_cycle alternates the final reset pi phase by 180 degrees between averaging sweeps (even reps, two-pulse/bath reset). It adds no RF pulse or programmed wait; verify population and phase witnesses.
