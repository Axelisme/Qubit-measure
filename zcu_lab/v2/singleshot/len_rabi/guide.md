## Behavior

Single-shot Length Rabi: sweeps the qubit-drive pulse length, preserves every raw IQ shot, and derives ground / excited / other population curves during live view and analysis. Runs on real hardware.

## Expects (parameter context)

Run freezes 'g_center' / 'e_center' / 'ge_radius' from resolved cfg, not live parameter context. Enter direct cfg values or optionally seed defaults with 'singleshot/ge' writeback. Missing or invalid cfg calibration fails before hardware. These values support live classification; the saved raw-IQ analysis jointly refits its calibration. Reads 'pi_len' to seed the sweep stop (4*pi_len when calibrated; fallback sweep 0.03–0.2 us); 'q_f' / 'qub_ch' to seed the qubit-drive defaults.

## Expects (ModuleLibrary)

Needs a qubit drive-pulse module (qub_pulse) and a readout module. Optional reset (disabled when no library entry exists).

## Typical writeback

When the joint fit is valid and its complete calibration is finite, proposes g_center, e_center, ge_radius, and confusion_matrix as four independent items. The four-item proposal is all-or-none.

## Recommended

Set Initial State to the predominant state before the swept drive pulse even when the first sweep point is nonzero. Fit phase offset allows a correction within +/-90 degrees of that initial-state direction; the extrapolated zero-length population can then differ from the pre-pulse population. It is disabled by default. Set calibration cfg directly or seed it with 'singleshot/ge'. A sweep spanning a few pi lengths captures a full oscillation. Review the measured population curves and overlaid joint-fit curves before applying all four calibration proposals.
