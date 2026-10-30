## Behavior

**Readers:** operator, agent. How to measure MIST populations versus preparation frequency from the GUI or MCP and interpret the results.

Sweep the preparation pulse frequency, then apply the optional pi pulse, the fixed MIST probe and readout. Reset precedes each sequence when enabled. This differs from singleshot/mist/freq, which sweeps the MIST probe itself.

The live plot shows Ground, Excited and Other versus preparation frequency in MHz. Other is the unclassified remainder; calibrated leakage requires separate characterization. Analyze reads an optional confusion matrix from the saved readout cfg and publishes a population plot for operator interpretation. Calibration parameters stay unchanged.

## Starting values

Open and Reset take a snapshot of aliases, parameters and module choices. The preparation pulse starts from the qub_probe template. Its sweep spans `$qubit.freq +/- 1.5 * $qubit.kappa` with 51 points. The fixed MIST probe starts from res_probe at `$resonator.freq`. Check both pulse gains and lengths before Run.

Optional reset and pi pulse use `$qubit.module.reset` and `$qubit.module.x180`. Missing optional slots start disabled. Readout uses `$resonator.module.readout` when available, otherwise the readout template. Missing wiring or frequencies remain Invalid. Run also requires a complete circle classifier with radius on every readout ADC.

Classification uses the readout's circle classifier under Readout → Advanced (per tone for Mux). Run requires classifiers on every declared ADC and positive radii; missing radius prompts singleshot GE Post. Linear has no Other state and is rejected. Analyze corrects only with the saved readout cfg's matrix. Historical cfg without a classifier or matrix skips correction.

Relax delay starts at five times the captured T1. Missing or null T1 uses a 30 us time scale, giving 150 us. An invalid stored T1 leaves the required delay empty. Defaults remain editable and separate from calibration writeback. Reps starts at 1000 and rounds at 10.

## Operator workflow

1. Select the qubit and resonator aliases, then Open singleshot/mist/pre_freq.
2. Check preparation, pi, MIST probe and readout settings. Enable the pi pulse only when it is part of the intended sequence.
3. Set the preparation-frequency span and shot count. Reset rebuilds defaults from a fresh source snapshot.
4. Preview the configuration, then Run. Analyze shows the corrected populations when a confusion matrix is available.
5. Save the run and figures. Saved data retain the rounded preparation-frequency axis and measured G/E/Other fractions shaped Nfreq by 3.

Analyze reads the confusion matrix from that result's saved readout cfg. Later alias, library or form edits do not change its correction.

Load displays native populations offline and keeps the current form unchanged. Correction uses only the file's saved readout classifier. Missing historical cfg, classifier or matrix skips correction. No live point supplies a fallback.

The operator chooses pulse conditions and interprets the population curves manually; calibration parameters stay unchanged.

## Saved counts

Save keeps `counts` shaped `(Nfreq, 3)` in `[g,e,other]` order, with
`population_states=[0,1,2]`. Other counts shots outside the accepted circles
or tied between them. It is not a calibrated leakage population.
Each point sums to the shots acquired across completed rounds. Live plots and
Analyze divide each point by its total before displaying populations or applying
the saved confusion matrix. Sweep coordinates and units are unchanged. Cfg schema
remains 3.0.

Old files storing `signals` instead of `counts` are unsupported. Load reports
the missing `counts` variable. Acquire a new run; Load does not convert stored
fractions to counts.

**Last updated:** 2026-10-29
