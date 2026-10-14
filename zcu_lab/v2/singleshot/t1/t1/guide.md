## Behavior

Single-shot T1: applies a π pulse, waits for a variable delay, reads out, and classifies each shot in-program against the |g>/|e> IQ-cluster centres. Repeats from both |g> and |e> initial states (Branch). Plots the ground / excited / other populations versus delay time with dual-transition-rate fits for T1 and T1_b. Runs on real hardware.

## Expects (parameter context)

Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved cfg, not live parameter context. Enter direct cfg values or optionally seed defaults with 'singleshot/ge' writeback. Missing or invalid cfg calibration fails before hardware. Optionally reads 'confusion_matrix' to readout-correct populations at analyze time; 't1' to seed the sweep stop (5*t1; fallback 500 us) and relax delay (5*t1; fallback 100 us); 'q_f' / 'qub_ch' for the pi pulse; 'r_f' / 'res_ch' / 'ro_ch' / 'timeFly' for readout.

## Expects (ModuleLibrary)

Needs a qubit pi-pulse module and a readout module. Optional reset (disabled when no library entry exists).

## Typical writeback

No writeback — T1 is shown in the plot title only. Use 'singleshot/t1_tone' if you need the T1 scalar in the parameter context.

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'. A delay sweep reaching ~5*T1 lets the decay flatten; with no prior 't1', the sweep spans 0–500 us. Set 'uniform=True' to sweep linearly; leave False to cluster more points along the expected exponential decay while preserving the configured start/stop window and point count.
