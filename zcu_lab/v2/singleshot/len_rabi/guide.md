## Behavior

Single-shot Length Rabi: sweeps the qubit-drive pulse length, preserves every raw IQ shot, and derives ground / excited / other population curves during live view and analysis. Runs on real hardware.

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay starts at 50.5 us. Length sweep is 0.03 us to 4 * `$qubit.module.x180.waveform.length.` Inline qubit gain starts at 1.0. Classification seeds `$qubit.ext.readout_cal.g_center`/e_center/radius; missing calibration stays Invalid. Inline templates read `$qubit.freq`, `$qubit.wiring.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Missing required sources remain Invalid; no numeric fallback is supplied.

## Expects (ModuleLibrary)

Qubit pulse starts from qub_probe. Reset uses optional `$qubit.module.reset`; readout uses `$qubit.module.readout` or readout template. Sweep-overwritten pulse values remain editable and library keys stay selectable. Run uses this fixed capture.

## Typical writeback

When the joint fit is valid and its complete calibration is finite, proposes `$qubit.ext.readout_cal.g_center`, `$qubit.ext.readout_cal.e_center`, `$qubit.ext.readout_cal.radius`, and `$qubit.ext.readout_cal.confusion_matrix` as four independent items. The four-item proposal is all-or-none.

## Recommended

Set Initial State to the predominant state before the swept drive pulse even when the first sweep point is nonzero. Fit phase offset allows a correction within +/-90 degrees of that initial-state direction; the extrapolated zero-length population can then differ from the pre-pulse population. It is disabled by default. Set calibration cfg directly or seed it with 'singleshot/ge'. A sweep spanning a few pi lengths captures a full oscillation. Review the measured population curves and overlaid joint-fit curves before applying all four calibration proposals.
