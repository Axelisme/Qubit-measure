## Behavior

Single-shot T1-with-tone, probe-gain sweep: at each probe gain it sweeps the wait-plus-probe length from both |g> and |e> initial states (Branch), classifies each shot in-program against the |g>/|e> centres, fits the dual transition rates per gain, and renders the 3×3 rate landscape over (gain, time). Runs on real hardware.

## Expects (parameter context)

Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved cfg, not live parameter context. Enter direct cfg values or optionally seed defaults with 'singleshot/ge' writeback. Missing or invalid cfg calibration fails before hardware. Optionally reads 'confusion_matrix' (readout correction) and 'ac_stark_coeff' (rescales the gain axis to photon number) at analyze time; 't1' to seed the length sweep stop; 'readout_f' or 'r_f' plus 'res_ch' seed the probe tone.

## Expects (ModuleLibrary)

Needs a qubit pi-pulse module, a probe-tone pulse module, and a readout module. Optional reset (disabled when no library entry exists).

## Typical writeback

No writeback — the rate landscape is read off the grid by eye.

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'. Keep the gain sweep coarse (the inner length sweep multiplies the run time). Provide 'ac_stark_coeff' for a photon-number x-axis. 'uniform=True' (default) uses a linear length sweep; set False to cluster points along the expected exponential decay while preserving the configured window and count.
