## Behavior

Hardware gain sweep preserving every raw IQ shot. Live view classifies Ground / Excited / Other; analysis shares the Len Rabi joint IQ fit with no decay and fixed zero phase.

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay starts at 50.5 us. Gain sweep starts at -0.3 and ends at 4 * `$qubit.module.x180.gain.` Inline qubit gain starts at 1.0. Classification seeds `$qubit.ext.readout_cal.g_center`/e_center/radius; missing calibration stays Invalid. Inline templates read `$qubit.freq`, `$qubit.wiring.L01.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Missing required sources remain Invalid; no numeric fallback is supplied.

## Expects (ModuleLibrary)

Qubit pulse starts from qub_probe. Reset uses optional `$qubit.module.reset`; readout uses `$qubit.module.readout` or readout template. Sweep-overwritten pulse values remain editable and library keys stay selectable. Run uses this fixed capture.

## Typical writeback

Reports pi/pi2 gain, frequency and amplitude. A valid joint fit proposes `$qubit.ext.readout_cal.g_center`, `$qubit.ext.readout_cal.e_center`, `$qubit.ext.readout_cal.radius` and `$qubit.ext.readout_cal.confusion_matrix` using the same calibration gate as Len Rabi.

## Recommended

Sweep a full Rabi period. Initial State describes the predominant state before the drive, at zero gain, even if the first gain is nonzero. Reps are shots per gain per round; rounds concatenate shots and update live populations. Population-only files cannot supply the raw IQ required for joint fitting.
