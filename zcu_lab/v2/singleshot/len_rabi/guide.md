## Behavior

Single-shot Length Rabi: sweeps the qubit-drive pulse length, preserves every raw IQ shot, and derives ground / excited / other population curves during live view and analysis. Runs on real hardware.

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay starts at 5*T1. Length sweep is 0.03 us to 4 * `$qubit.module.x180.waveform.length.` Classification seeds `$qubit.ext.readout_cal.g_center`/e_center/radius; missing calibration stays Invalid. Inline templates read `$qubit.freq`, `$qubit.wiring.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Other missing required sources remain Invalid.

Both Length Rabi forms sweep from 0.03 us to 4*pi_len. Missing/null pi_len, including arb without a persistent scalar length, uses a 1 us stop; stored invalid pi_len leaves stop empty. Valid pi_len is a finite positive non-bool number. Existing x180 stays a Slot with its gain; without x180 both forms use qub_probe with gain=1. No asset is read to obtain a runtime length.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Qubit pulse follows the shared Rabi source policy above, using x180 when present and qub_probe only when absent. Reset uses optional `$qubit.module.reset`; readout uses `$qubit.module.readout` or readout template. Sweep-overwritten pulse values remain editable and library keys stay selectable. Run uses this fixed capture.

## Typical writeback

When the joint fit is valid and its complete calibration is finite, proposes `$qubit.ext.readout_cal.g_center`, `$qubit.ext.readout_cal.e_center`, `$qubit.ext.readout_cal.radius`, and `$qubit.ext.readout_cal.confusion_matrix` as four independent items. The four-item proposal is all-or-none.

## Recommended

Set Initial State to the predominant state before the swept drive pulse even when the first sweep point is nonzero. Fit phase offset allows a correction within +/-90 degrees of that initial-state direction; the extrapolated zero-length population can then differ from the pre-pulse population. It is disabled by default. Set calibration cfg directly or seed it with 'singleshot/ge'. A sweep spanning a few pi lengths captures a full oscillation. Review the measured population curves and overlaid joint-fit curves before applying all four calibration proposals.
