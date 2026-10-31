# Heralded four-branch joint readout

**Readers:** operator, agent. Run `two_qubit/joint_readout_heralded` and read its accepted terminal distribution and herald pass fraction.

## Before run

Follow the [ordinary joint readout setup](guide.md#before-run) for captured F1/F2 roles, full mux ordering, calibrated Xπ pulses and circle classifiers with GE Post radii. Both versions reject missing radius and non-circle classifiers before Run. Review the full readout and reset settings against hardware limits.

This is a separate GUI and registry entry from `two_qubit/joint_readout`. Each has its own cfg and Result. `reset` starts as None. Supply a calibrated reset only when wanted. Heralding itself does not reset either qubit.

## Sequence and saved counts

Each shot emits optional reset, a full mux read tagged `herald`, the nominal I/Xπ preparation, and a full mux read tagged `terminal`. Both reads emit the same mux module with the same gain, integration window and classifiers. Readout post_delay supplies the wait after herald. There is no extra herald-delay field.

Branch order is 00, 01, 10, 11, with F1 first. Branch 11 applies F1 Xπ then F2 Xπ. `reps` counts shots per round per branch. `rounds` adds completed rounds during acquisition, without a host round axis.

The processor saves all 81 joint classes as float64 `sums` shaped `(4,3,3,3,3)`. After branch, the axes are herald F1, herald F2, terminal F1, terminal F2. Each class axis is g/e/other. Counts remain unnormalized, including rejected herald classes. Native persistence generates class coordinates and retains the actual run cfg. No per-shot IQ survives in Result.

The live plot selects the herald g,g slice and shows its terminal P(g,g). Cancel keeps completed rounds and leaves unmeasured branches NaN. The caller still reports acquisition errors as failures. Old files without summed counts fail Load; no compatibility conversion runs.

## Analysis

FIT selects terminal counts where herald classified both F1 and F2 as g. It then uses the same 4×9 count normalization as ordinary joint readout. Terminal other remains in the denominator. Zero accepted shots give NaN terminal probabilities. A zero fraction does not become a synthetic g,g distribution.

A second figure shows all nine herald proportions for each branch. Its g,g cell is the **pass fraction**, accepted shots divided by all acquired shots for that branch. Pass fraction is not ground-state purity. Classification errors and initial populations both affect acceptance. Terminal preparation and readout errors remain after selection.

The analysis summary returns the terminal counts and matrix, herald proportions, and pass fraction. FIT offers no writeback or correction. Analyze reads saved counts, not current library settings. Compare with the ordinary version to diagnose thermal contributions, rather than treating the difference as a gate-fidelity measurement.

Compile-only checks do not verify acquisition, save/restore, physical populations or plot appearance. Those need the separate offline and control-chain validation steps.
