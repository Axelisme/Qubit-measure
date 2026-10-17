## Behavior

Hardware gain sweep preserving every raw IQ shot. Live view classifies Ground / Excited / Other; analysis shares the Len Rabi joint IQ fit with no decay and fixed zero phase.

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay starts at 5*T1. Gain sweep runs from -0.3 to 1.0. Classification seeds `$qubit.ext.readout_cal.g_center`/e_center/radius; missing calibration stays Invalid. Inline templates read `$qubit.freq`, `$qubit.wiring.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Other missing required sources remain Invalid.

Both Amp Rabi forms sweep gain from -0.3 to 1.0 regardless of calibration. For x180 with a persistent scalar waveform length, the inline copy starts at length = 1.5*pi_len*pi_gain us. pi_len must be a finite positive non-bool number and pi_gain must be finite, non-bool and in (0, 1]. Missing/null or invalid leaves leave length required and empty. Finite invalid gain stays visible; nonfinite gain leaves both gain and length empty. Without x180, qub_probe starts at gain=1 and length=1 us. Arb x180 keeps its original Slot, shape and data without copying or overriding length or reading assets; change waveform first to edit duration.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Qubit pulse follows the shared Rabi source policy above, using x180 when present and qub_probe only when absent. Reset uses optional `$qubit.module.reset`; readout uses `$qubit.module.readout` or readout template. Sweep-overwritten pulse values remain editable and library keys stay selectable. Run uses this fixed capture.

## Typical writeback

Reports pi/pi2 gain, frequency and amplitude. A valid joint fit proposes `$qubit.ext.readout_cal.g_center`, `$qubit.ext.readout_cal.e_center`, `$qubit.ext.readout_cal.radius` and `$qubit.ext.readout_cal.confusion_matrix` using the same calibration gate as Len Rabi.

## Recommended

Sweep a full Rabi period. Initial State describes the predominant state before the drive, at zero gain, even if the first gain is nonzero. Reps are shots per gain per round; rounds concatenate shots and update live populations. Population-only files cannot supply the raw IQ required for joint fitting.
