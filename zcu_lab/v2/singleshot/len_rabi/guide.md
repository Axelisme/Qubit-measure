## Behavior

**Readers:** operator, agent. How to run single-shot length Rabi calibration from the GUI or MCP and interpret the results.

Single-shot Length Rabi: sweeps the qubit-drive pulse length, preserves every raw IQ shot, and derives ground / excited / other population curves during live view and analysis. Runs on real hardware.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay starts at 5*T1. Length sweep is 0.03 us to 4 * `$qubit.module.x180.waveform.length`.

Classification starts from `$qubit.ext.readout_cal.g_center`, `e_center` and `radius`. Missing calibration leaves these fields Invalid.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

The length sweep runs from 0.03 us to 4*pi_len. A missing or null stored pi_len uses a 1 us stop, including for arbitrary waveforms whose scalar length is absent. An invalid stored pi_len leaves the required stop empty for correction. A valid pi_len is a finite positive number, with booleans treated as invalid.

An existing x180 stays selected through its module slot and keeps its gain. If x180 is absent, the qub_probe template starts at gain=1. The default stop depends only on the stored scalar length.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

The qubit pulse uses the x180 module when present and the qub_probe template when absent, with the starting values described above. Reset uses optional `$qubit.module.reset`; readout uses `$qubit.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Reset snapshot.

## Typical writeback

When the joint fit is valid and its complete calibration is finite, proposes `$qubit.ext.readout_cal.g_center`, `$qubit.ext.readout_cal.e_center`, `$qubit.ext.readout_cal.radius`, and `$qubit.ext.readout_cal.confusion_matrix` as four separately applicable items. All four proposals appear together when the complete calibration passes validation; otherwise the proposal list stays empty.

## Recommended

Set Initial State to the predominant state before the swept drive pulse even when the first sweep point is nonzero. Fit phase offset allows a correction within +/-90 degrees of that initial-state direction; the extrapolated zero-length population can then differ from the pre-pulse population. It is disabled by default. Set the classification calibration directly or seed it with 'singleshot/ge'. A sweep spanning a few pi lengths captures a full oscillation. Review the measured population curves and overlaid joint-fit curves before applying all four calibration proposals.

**Last updated:** 2026-10-21
