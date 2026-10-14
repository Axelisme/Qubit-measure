## Behavior

MIST probe power × frequency landscape: drives a probe pulse while sweeping its gain and frequency (2D), classifying each shot in-program against the |g>/|e> single-shot centres, and plotting the ground/excited/other populations as 2D maps over (gain, freq). Runs on real hardware; the result is already populations (no per-point fit).

## Expects (parameter context)

Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved cfg, not live parameter context. Enter direct cfg values or optionally seed defaults with 'singleshot/ge' writeback. The run classifies each shot using these values; missing or invalid cfg calibration fails before hardware. Optionally reads 'confusion_matrix' for readout correction at analyze time, and 't1' to set the relax delay; 'readout_f' or 'r_f' plus 'rf_w' / 'res_ch' seed the probe drive and frequency sweep.

## Expects (ModuleLibrary)

Needs a probe pulse and a readout module. Optionally references a calibrated reset and an init pulse — both disabled when no library entry exists.

## Typical writeback

No writeback — the population landscapes are read off the plot by eye.

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'. Sweep the probe gain across the MIST onset and the frequency around the qubit/resonator line of interest.
