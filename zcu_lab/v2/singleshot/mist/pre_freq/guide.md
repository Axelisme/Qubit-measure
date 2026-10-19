## Behavior

Sweep the preparation pulse frequency, then apply the optional pi pulse, the fixed MIST probe and readout. Reset precedes each sequence when enabled. This differs from singleshot/mist/freq, which sweeps the MIST probe itself.

The live plot shows Ground, Excited and Other versus preparation frequency in MHz. Other is the unclassified remainder, not calibrated leakage. Analyze applies the optional confusion matrix from the fixed run capture and publishes a population plot. It does not fit a resonance or propose writeback.

## Starting values

Open and Reset capture aliases, parameters and the module library. The preparation pulse starts from the qub_probe template. Its sweep spans `$qubit.freq +/- 1.5 * $qubit.kappa` with 51 points. The fixed MIST probe starts from res_probe at `$resonator.freq`. Check both pulse gains and lengths before Run.

Optional reset and pi pulse use `$qubit.module.reset` and `$qubit.module.x180`. Missing optional slots start disabled. Readout uses `$qubit.module.readout` when available, otherwise the readout template. Missing required wiring, frequencies or classification calibration remain Invalid. Supply them before previewing a complete cfg or running.

Classification uses `$qubit.ext.readout_cal.g_center`, `e_center` and `radius`. Calibrate these with singleshot/ge at the same readout settings. The centres must be finite and distinct, and radius must be positive and finite.

Relax delay starts at five times the captured T1. Missing or null T1 uses a 30 us time scale, giving 150 us. An invalid stored T1 leaves the required delay empty. Defaults remain editable and do not update calibrations. Reps starts at 10000 and rounds at 1.

## Operator workflow

1. Select the qubit and resonator aliases, then Open singleshot/mist/pre_freq.
2. Check preparation, pi, MIST probe and readout settings. Disable the pi pulse if the intended sequence does not use it.
3. Set the preparation-frequency span and shot count. Reset rebuilds defaults from a fresh source capture.
4. Preview cfg, then Run. Analyze shows the corrected populations when a confusion matrix is available.
5. Save the run and figures. Saved data retain the rounded preparation-frequency axis and G/E fractions shaped Nfreq by 2. Other remains derived.

Run, Analyze and Save use the run's fixed sources, not later alias or form edits. Load supports the native population format without hardware and does not replace the current form. No automatic recipe or writeback is provided because choosing pulse conditions and interpreting the population curves require operator review.
