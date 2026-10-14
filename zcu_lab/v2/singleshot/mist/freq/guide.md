## Behavior

MIST probe-frequency sweep: drives a probe pulse whose frequency is swept and classifies each shot in-program against the |g>/|e> single-shot centres, plotting the ground/excited/other populations versus probe frequency. Runs on real hardware; the result is already populations (no per-point fit).

## Expects (parameter context)

Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved cfg, not live parameter context. Enter direct cfg values or optionally seed defaults with 'singleshot/ge' writeback. The run classifies each shot using these values; missing or invalid cfg calibration fails before hardware. Optionally reads 'confusion_matrix' (the GE 3x3 matrix) to readout-correct the populations at analyze time, and 't1' to set the relax delay; 'readout_f' or 'r_f' plus 'rf_w' / 'res_ch' seed the probe drive and frequency sweep.

## Expects (ModuleLibrary)

Needs a probe pulse and a readout module. Optionally references a calibrated reset and an init pulse — both disabled when no library entry exists.

## Typical writeback

No writeback — the population curves are read off the plot by eye.

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'. A frequency span around the qubit drive captures the MIST response; use a large enough shot count (reps) for clean populations.
