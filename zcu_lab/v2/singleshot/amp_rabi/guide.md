## Behavior

Hardware gain sweep preserving every raw IQ shot. Live view classifies Ground / Excited / Other; analysis shares the Len Rabi joint IQ fit with no decay and fixed zero phase.

## Expects (parameter context)

Freezes g_center / e_center / radius from resolved cfg for live classification. Set calibration directly or seed defaults with singleshot/ge writeback; invalid calibration fails before hardware. Analysis jointly refits readout calibration without an external confusion matrix. Reads pi_gain for the sweep range.

## Expects (ModuleLibrary)

Needs qub_pulse and readout; upstream reset is optional.

## Typical writeback

Reports pi/pi2 gain, frequency and amplitude. A valid joint fit proposes g_center, e_center, ge_radius and confusion_matrix using the same calibration gate as Len Rabi.

## Recommended

Sweep a full Rabi period. Initial State describes the predominant state before the drive, at zero gain, even if the first gain is nonzero. Reps are shots per gain per round; rounds concatenate shots and update live populations. Population-only files cannot supply the raw IQ required for joint fitting.
