## Behavior

Single-shot classification check: takes 'shots' single-shot readouts of the prepared state and draws the IQ scatter with the |g>/|e> discrimination circles overlaid, reporting the fraction classified ground / excited / other. Runs on real hardware; the run itself needs no centres, but the analysis classifies the scatter against them.

## Expects (parameter context)

REQUIRES the single-shot discrimination calibration in the parameter context for ANALYSIS — run 'singleshot/ge' first and apply its writeback so 'g_center' / 'e_center' / 'ge_radius' are present; the analyze classifies the scatter against them and fast-fails if any is missing. Also reads 't1' to set the relax delay; 'q_f' / 'qub_ch' seed the probe drive.

## Expects (ModuleLibrary)

Needs a probe pulse and a readout module. Optionally references a calibrated reset and an init pulse — both disabled when no library entry exists.

## Typical writeback

No writeback — the check is a visual diagnostic of the existing single-shot discrimination.

## Recommended

Run after 'singleshot/ge'. Use a large 'shots' (~5000+) for a well-sampled scatter; a tight cluster inside the matching circle indicates good discrimination.
