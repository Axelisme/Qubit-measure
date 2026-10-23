## Behavior

**Readers:** operator, agent. How to calibrate ground/excited single-shot readout from the GUI or MCP and interpret the results.

Single-shot ground/excited readout: measures with the probe pi-pulse off and on, takes 'shots' single-shot readouts of each, and fits the two IQ clusters to extract the assignment fidelity, rotation angle and threshold. Runs on real hardware with rounds=1 and reps=shots. Each run acquires readout shots twice, with the probe off and on.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay is 5 * `$qubit.t1`. The probe template uses `$qubit.freq` and `$qubit.wiring.ch`.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Uses `$qubit.module.x180` for optional init and required probe, `$qubit.module.reset` for optional reset, and `$qubit.module.readout` for readout. Missing required slots use pi_pulse/readout templates; missing optional slots stay disabled. Run uses the Open/Reset snapshot.

## Assignment and model quality

Fidelity is the empirical, equal-weight assignment of the two prepared labels at
the best projected-IQ histogram boundary. Probe off/on are preparation labels,
not guaranteed pure g/e states. Thermal mixing and imperfect pulses remain in
this score. A threshold can also exploit small differences inside one cluster;
a high prepared-label score does not establish pure-state readout fidelity.

Initial populations come from the fitted histogram model. Post-analysis instead
classifies shots by nearest-center regions and a radius, then solves the fitted
population matrix to remove preparation mixing. It does not reuse the primary
histogram threshold. These outputs need not equal the prepared-label score.
Low joint r2 means the histogram model does not describe the observations;
then fitted populations and population-corrected post results are not reliable
calibration evidence. Inspect raw clusters, residuals and uncertainty before
using centers, width or the confusion matrix.

## Typical writeback

Primary candidates are initially selected only when joint r2 is at least 0.9,
normalized residual RMS at most 0.1 and relative width error at most 0.5.
Post radius/confusion also require a finite condition number at most 10.
Required missing/nonfinite metrics or their invalid issues leave candidates
unselected with notes. Unrelated fixed-zero parameter issues do not reject them.
You may select a candidate manually after inspecting the model and raw IQ.

Analysis and writeback use the fixed parameter source of the selected Run or Load result. Run keeps the source used for that measurement.

For a file from the same opened entry, Load reads the current persisted values at the file's historical point, using its role assignments. It does not use the tab's Open/Reset aliases or restore an unsaved historical parameter version. Files from another entry or without a point have no parameter source for analysis or writeback. Analysis that needs no such source remains available.

Writeback targets `$qubit.ext.readout_cal` for the source qubit when that source is available. Primary analysis proposes `fidelity`, `sigma`, `g_center` and `e_center`. Post-Analysis proposes `radius` and `confusion_matrix`. The centres are complex IQ coordinates and the confusion matrix is a nested list. Existing proposals stay attached to that qubit when point or focus changes.

## Recommended

Set Initial State to the predominant state after reset/init and before the probe pi-pulse.
This labels the two acquisitions; each can contain a mixture of states.
Use a large 'shots' (~1e5) so the IQ histograms are well sampled.
The default analysis method is 'pca'.
Analysis also exposes histogram log scale, T1 alignment, and an optional shared length ratio.
Run once the qubit pi-pulse and the readout are both calibrated.
Run and Analyze each show one IQ scatter, with blue g-preparation and red e-preparation shots.
The live labels mean probe off/on; Analyze maps them using Initial State.
Histogram fitting remains internal, so histogram log scale does not change the scatter.
Two separated IQ clouds indicate readout separation.
Use Post-Analysis to inspect the classified shots and 3x3 confusion diagnostic derived from the primary fit.

**Last updated:** 2026-10-21
