## Behavior

Single-shot T1-with-tone: applies a π pulse and a simultaneous probe tone during the wait, sweeps the wait-plus-probe length, and classifies each shot in-program against the |g>/|e> IQ-cluster centres. Plots the ground / excited / other populations versus delay time with dual-transition-rate fits for T1 and T1_b. The fitted T1 value is written back to 't1_with_tone' in the parameter context. Runs on real hardware.

## Expects (parameter context)

Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved cfg, not live parameter context. Enter direct cfg values or optionally seed defaults with 'singleshot/ge' writeback. Missing or invalid cfg calibration fails before hardware. Optionally reads 'confusion_matrix' to readout-correct populations at analyze time; 't1_with_tone' or 't1' to seed the sweep stop (5*t1_with_tone when present, else 5*t1; fallback 500 us); 't1' seeds relax delay (5*t1; fallback 100 us); 'q_f' / 'qub_ch' for the pi pulse; 'readout_f' or 'r_f' plus 'res_ch' seed the probe tone; 'r_f' / 'res_ch' / 'ro_ch' / 'timeFly' for readout.

## Expects (ModuleLibrary)

Needs a qubit pi-pulse module, a probe-tone pulse module, and a readout module. Optional reset and optional init pulse (both disabled when no library entry exists).

## Typical writeback

Proposes the fitted T1 relaxation time under the probe tone into parameter context 't1_with_tone' (us).

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'. Use 'uniform=False' (default) to cluster points along the expected exponential decay while preserving the configured window and point count; use 'uniform=True' for a linear sweep. The probe-tone gain and frequency are set inside the probe_pulse module.
