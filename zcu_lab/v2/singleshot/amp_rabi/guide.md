## Behavior

**Readers:** operator, agent. How to run single-shot amplitude Rabi calibration from the GUI or MCP and interpret the results.

The hardware gain sweep preserves every raw IQ shot. Live view classifies Ground / Excited / Other. Analysis uses a joint IQ fit with constant oscillation amplitude and phase fixed at zero.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay starts at 5*T1. Gain sweep runs from -0.3 to 1.0.

Classification starts from `$qubit.ext.readout_cal.g_center`, `e_center` and `radius`. Missing calibration leaves these fields Invalid.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

The gain sweep starts at -0.3 and stops at 1.0 regardless of calibration. When the x180 module has a stored scalar waveform length, its editable pulse copy starts at length = 1.5*pi_len*pi_gain us. pi_len must be a finite positive number; pi_gain must be finite and in (0, 1]. Both values must be numeric, with booleans treated as invalid. Missing, null or invalid values leave the required length empty. A finite invalid gain stays visible for correction; a nonfinite gain leaves both gain and length empty.

If the x180 module is absent, the qub_probe template starts at gain=1 and length=1 us. An arbitrary-waveform x180 keeps the original module slot, waveform shape and data. Its duration stays with the selected waveform; change waveform first to edit duration.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

The qubit pulse uses the x180 module when present and the qub_probe template when absent, with the starting values described above. Reset uses optional `$qubit.module.reset`; readout uses `$resonator.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Reset snapshot.

## Typical writeback

Reports pi/pi2 gain, frequency and amplitude. A joint fit that passes the readout-calibration validity checks proposes `$qubit.ext.readout_cal.g_center`, `$qubit.ext.readout_cal.e_center`, `$qubit.ext.readout_cal.radius` and `$qubit.ext.readout_cal.confusion_matrix` with the same validity requirements as single-shot Length Rabi.

## Recommended

Sweep a full Rabi period. Initial State describes the predominant state before the drive, at zero gain, even if the first gain is nonzero. Reps are shots per gain per round; rounds concatenate shots and update live populations. Joint fitting requires a file containing raw IQ shots.

**Last updated:** 2026-10-21
