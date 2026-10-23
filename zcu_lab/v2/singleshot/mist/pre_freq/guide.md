## Behavior

**Readers:** operator, agent. How to measure MIST populations versus preparation frequency from the GUI or MCP and interpret the results.

Sweep the preparation pulse frequency, then apply the optional pi pulse, the fixed MIST probe and readout. Reset precedes each sequence when enabled. This differs from singleshot/mist/freq, which sweeps the MIST probe itself.

The live plot shows Ground, Excited and Other versus preparation frequency in MHz. Other is the unclassified remainder; calibrated leakage requires separate characterization. Analyze reads an optional confusion matrix from the selected result's parameter source and publishes a population plot for operator interpretation. Calibration parameters stay unchanged.

## Starting values

Open and Reset take a snapshot of aliases, parameters and module choices. The preparation pulse starts from the qub_probe template. Its sweep spans `$qubit.freq +/- 1.5 * $qubit.kappa` with 51 points. The fixed MIST probe starts from res_probe at `$resonator.freq`. Check both pulse gains and lengths before Run.

Optional reset and pi pulse use `$qubit.module.reset` and `$qubit.module.x180`. Missing optional slots start disabled. Readout uses `$qubit.module.readout` when available, otherwise the readout template. Missing required wiring, frequencies or classification calibration remain Invalid. Supply them before previewing a complete configuration or running.

Classification uses `$qubit.ext.readout_cal.g_center`, `e_center` and `radius` as starting values in the editable run configuration. Run classifies shots with those configured values, and saved data retain that configuration. Analyze separately reads the optional `$qubit.ext.readout_cal.confusion_matrix` from the selected result's parameter source to correct populations. Calibrate these values with singleshot/ge at the same readout settings. The centres must be finite and distinct, and radius must be positive and finite.

Relax delay starts at five times the captured T1. Missing or null T1 uses a 30 us time scale, giving 150 us. An invalid stored T1 leaves the required delay empty. Defaults remain editable and separate from calibration writeback. Reps starts at 1000 and rounds at 10.

## Operator workflow

1. Select the qubit and resonator aliases, then Open singleshot/mist/pre_freq.
2. Check preparation, pi, MIST probe and readout settings. Enable the pi pulse only when it is part of the intended sequence.
3. Set the preparation-frequency span and shot count. Reset rebuilds defaults from a fresh source snapshot.
4. Preview the configuration, then Run. Analyze shows the corrected populations when a confusion matrix is available.
5. Save the run and figures. Saved data retain the rounded preparation-frequency axis and G/E fractions shaped Nfreq by 2. Other remains derived.

For a Run result, Analyze reads the confusion matrix from the fixed parameter source used for that measurement. Later alias or form edits do not retarget that result.

Load displays the native population format offline and keeps the current form unchanged. For a file from the same opened entry, Load reads the current persisted values at the file's historical point, using its role assignments. It fixes those values as the analysis source, rather than restoring the confusion matrix used at the original measurement time. A calibration update at that point can therefore change the corrected populations after Load. The saved classification settings and G/E fractions stay unchanged. A file from another entry or without a point has no parameter source for confusion correction.

The operator chooses pulse conditions and interprets the population curves manually; calibration parameters stay unchanged.

**Last updated:** 2026-10-21
