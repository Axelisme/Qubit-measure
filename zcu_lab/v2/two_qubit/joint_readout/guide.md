# Four-branch joint readout

**Readers:** operator, agent. Run `two_qubit/joint_readout` and interpret its nine-class joint distribution.

## Before run

In the point's Aliases table, bind `qubit1` and `qubit2` to different components. Bind `mux` to their MuxGroup. Joint readout does not require a `coupler` binding. For an older point, add the two position aliases for the components you want to prepare first and second, then Reset the form. Notebook uses the same built-in position names, but its bindings do not populate the GUI's Aliases table.

Open and Reset find both components in `$mux.qubits`. Their positions in that list are the zero-based tone indices. Keep the complete readout in the same order, including any tones outside Qubit1 and Qubit2. If the captured readout has a different number of tones, Open or Reset reports the mismatch. The form hides the indices. To verify them, inspect `qubit1_tone` and `qubit2_tone` in the saved run cfg. Refresh preserves authored inputs, so use Reset after changing bindings or calibration sources.

Provide calibrated `$qubit1.module.x180` and `$qubit2.module.x180` pulses on different charge generators. Readout starts from `$mux.module.readout` or a blank mux input if the slot is absent. Confirm generator, mixer, ADC channels, integration window, trigger timing and headroom before Run.

The Qubit1 and Qubit2 tones each need a circle classifier **with radius**. Run mux singleshot GE and apply GE Post under the actual readout conditions. Missing calibration, linear or centroid3 mode, and missing radius make the cfg invalid with a GE Post prompt. Other tones need no classifier for this experiment. Recalibrate after changing readout settings.

`reset` starts as None. Leave it None to omit reset, or supply a calibrated reset configuration to emit it at the start of every shot. Set DC devices externally before Run.

## Sequence and acquisition

Each shot starts with optional reset, applies its nominal I/Xπ preparation, then reads the full mux once with tag `terminal`. One program sweeps two hardware axes, `prepared_qubit1` outside `prepared_qubit2`. Each axis uses 0 for g (I) and 1 for e (Xπ), giving row order gg, ge, eg, ee.

The program forks directly into Qubit1 and Qubit2 timelines after optional reset, without an extra delay_by_timeline. The forked timelines inherit the reset end. Each timeline has an aligned branch that emits its Xπ only for case 1; case 0 is empty. Both Xπ pulses start together for ee. The timelines join before terminal readout, so all preparations have the same readout start time. Without reset, preparations act on the thermal ensemble.

`reps` is shots per acquisition round per branch. `rounds` accumulates completed rounds at acquisition. Initial and relaxation delays use µs.

The reducer classifies Qubit1 and Qubit2 on each shot and sums the nine joint classes. Each position has g, e and other. Circle boundaries and equidistant points count as other. The experiment does not save per-shot IQ and does not multiply separate qubit probabilities.

The live plot shows terminal P(g,g) from accumulated counts for each nominal branch. Cancel preserves completed rounds. Unmeasured branches stay NaN. A failed acquisition remains an operation failure even when partial counts survive.

## Saved data and analysis

The HDF5 file stores unnormalized float64 counts in `/vars/sums` with shape `(2,2,3,3)`. Its first two axes are `Prepared Qubit1 (0=g, 1=e)` and `Prepared Qubit2 (0=g, 1=e)`, each with coordinates `[0, 1]`. The last two axes are `terminal_qubit1` and `terminal_qubit2`; coordinates `[0, 1, 2]` mean g, e and other. The file also stores the cfg used for the run. File size does not grow with shot count.

FIT shows a 4×9 count and row-normalized probability matrix. Columns are Qubit1-major joint classes from g,g through other,other. Every class, including other, contributes to the denominator. Zero-count and unmeasured rows give NaN probabilities. Partial acquisitions can still provide distributions from their completed rounds. Each row label shows its acquired shot count.

The matrix includes initial populations, preparation errors and readout effects. It is not a pure-state assignment matrix. FIT offers no correction or writeback. Do not infer gate fidelity or invert this matrix without separate calibration evidence.

Load rejects files whose `/vars/sums` axis names do not match, including files saved with the former position-axis names. Keep the original file unchanged and open it with the corresponding older software version. Load also rejects old raw-IQ files without `sums` and files with a single preparation axis. Analyze uses the saved counts without looking up the current library or cfg.

## Select shots with mux herald

Choose `reset/mux_herald` in Reset. Use the same mux readout as the terminal
readout. Set `keep: {'0': g, '1': g}` if Qubit1 and Qubit2 occupy tone positions 0 and 1.
Use their original tone positions from the full readout, not their ADC numbers.

Probabilities are conditional on passing herald. Divide each terminal class
count by the point's total count. Acceptance is the total divided by
reps × rounds, not ground-state purity. A point with no accepted shots has
NaN probabilities. Completed rounds remain usable when Run stops early.
Files saved under the former separate heralded experiment no longer load.

Continue physical calibration with the [FTF bring-up guide](../guide.md).

## Notebook

Use a validated `JointReadoutCfg` as `cfg`. Set `x180_qubit1` and `x180_qubit2`
to the calibrated pulses and supply `qubit1_tone` and `qubit2_tone` as indices
in the full readout. Supply the notebook's existing hardware handles and
device manager. `point_view` is the bound ParameterView for the
save destination. Run requires that environment; analyze uses the saved counts.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.joint_readout.core import JointReadoutExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(JointReadoutExp())
record = exp.run(cfg)
analysis = exp.analyze(None)
saved = exp.save_run(point_view, source=record)
```

Do not call core acquisition directly from the notebook. NotebookAdapter owns
the run record and the plots for each operation.
