# Heralded four-branch joint readout

**Readers:** operator, agent. Run `two_qubit/joint_readout_heralded` and read its accepted terminal distribution and herald pass fraction.

## Before run

Follow the [ordinary joint readout setup](guide.md#before-run) for captured F1/F2 roles, full mux ordering, calibrated Xπ pulses and circle classifiers with GE Post radii. Both versions reject missing radius and non-circle classifiers before Run. Review the full readout and reset settings against hardware limits.

This is a separate GUI and registry entry from `two_qubit/joint_readout`. Each has its own cfg and Result. `reset` starts as None. Supply a calibrated reset only when wanted. Heralding itself does not reset either qubit.

## Sequence and saved counts

Each shot emits optional reset, a full mux read tagged `herald`, the nominal I/Xπ preparation, and a full mux read tagged `terminal`. Both reads emit the same mux module with the same gain, integration window and classifiers. Readout post_delay supplies the wait after herald. There is no extra herald-delay field.

Branch order is 00, 01, 10, 11, with F1 first. Branch 11 applies F1 Xπ then F2 Xπ. A single program sweeps all four preparations using a hardware loop and four named cases. The gg case emits no Xπ. Each case keeps its own duration before terminal readout. `reps` counts shots per round per branch. `rounds` adds completed rounds during acquisition, without a host round axis.

The processor saves all 81 joint classes as float64 `sums` shaped `(4,3,3,3,3)`. Result stores the nominal `prepared_states=[0,1,2,3]` axis for gg, ge, eg, ee. After that axis, the axes are herald F1, herald F2, terminal F1, terminal F2. Each class axis is g/e/other. Counts remain unnormalized, including rejected herald classes. Native persistence generates class coordinates and retains the actual run cfg. No per-shot IQ survives in Result.

The live plot selects the herald g,g slice and shows its terminal P(g,g). Cancel keeps completed rounds and leaves unmeasured branches NaN. The caller still reports acquisition errors as failures. Old files without summed counts fail Load; no compatibility conversion runs.

## Analysis

FIT shows three 4×9 matrices, for herald g,g, herald e,e, and their combined counts. Each row names the actual state after the I/Xπ preparation, in order gg, ge, eg, ee. The g,g group uses nominal rows in that order. In the e,e group, the same pulses reach the complementary state, so nominal rows ee, eg, ge, gg map to actual rows gg, ge, eg, ee. Differences between the two routes to one actual state expose preparation-pulse errors.

The combined matrix adds the aligned raw counts before row normalization. It does not average group probabilities. Terminal other remains in every denominator. Zero selected shots and unmeasured rows give NaN probabilities. Each terminal row label shows its selected shot count. Partial acquisitions use completed rounds; they do not become complete operations.

A second figure shows all nine herald proportions for each nominal preparation, with its total acquired shot count. Herald ge, eg and any other classes remain saved but do not enter the three terminal matrices. Its g,g cell is the **pass fraction**, accepted shots divided by all acquired shots for that branch. Pass fraction is not ground-state purity. Classification errors and initial populations both affect acceptance. Terminal preparation and readout errors remain after selection.

The analysis returns gg, ee and combined counts and matrices, selected shot totals, total acquired shots, herald proportions, and pass fraction. FIT offers no writeback or correction. Analyze reads saved counts, not current library settings. Compare with the ordinary version to diagnose thermal contributions, rather than treating the difference as a gate-fidelity measurement.

Compile-only checks do not verify acquisition, save/restore, physical populations or plot appearance. Those need the separate offline and control-chain validation steps.

## Notebook

Use a validated `JointReadoutHeraldedCfg` as `cfg`. Supply the notebook's existing
hardware handles and device manager. `point_view` is the bound PointView for the
save destination. Run requires that environment; analyze uses the saved counts.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.joint_readout.core_heralded import JointReadoutHeraldedExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(JointReadoutHeraldedExp())
record = exp.run(cfg)
analysis = exp.analyze(None)
saved = exp.save_run(point_view, source=record)
```

Do not call core acquisition directly from the notebook. NotebookAdapter owns
the run record and the plots for each operation.
