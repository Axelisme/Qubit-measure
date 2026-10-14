## Behavior

Single-shot ground/excited readout: measures without and with the probe pi-pulse, takes 'shots' single-shot readouts of each, and fits the two IQ clusters to extract the assignment fidelity, rotation angle and threshold. Runs on real hardware; the domain forces rounds=1 and reps=shots, running the readout twice (probe off / on) internally.

## Expects (parameter context)

Defaults use the tab's fixed qubit and referenced resonator. Relax delay is 5 * qubit.t1 (fallback 100 us). Inline probe reads qubit.freq and qubit.wiring.L01.ch. Inline pulse-readout reads resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us (fallback 0.55 us). Missing or null calibration uses fallback; unbound roles remain Invalid until supplied.

## Expects (ModuleLibrary)

Uses captured qubit.module.x180 for init and probe and qubit.module.readout for pulse-readout. Missing or dangling probe/readout slots seed fresh inline defaults. Optional init and qubit.module.reset remain disabled when absent. Verify the inline pulse and readout timing before Run.

## Typical writeback

Writes to the captured qubit.ext.readout_cal: primary proposes fidelity, sigma, g_center and e_center; post-analysis proposes radius and the nested-list confusion_matrix. Centers remain native complex proposals; accepted values use the shared __complex__ tag. Changing point or focus does not retarget an existing proposal.

## Recommended

Set Initial State to the predominant state after reset/init and before the probe pi-pulse. This labels the two acquisitions, not a pure-state prior. Use a large 'shots' (~1e5) so the IQ histograms are well sampled; the default analysis backend is 'pca'. Analysis also exposes histogram log scale, T1 alignment, and an optional shared length ratio; advanced population priors remain internal. Run once the qubit pi-pulse and the readout are both calibrated — a clean two-cluster IQ scatter indicates good discrimination. Use Post-Analysis to inspect the classified shots and 3x3 confusion diagnostic derived from the primary fit.
