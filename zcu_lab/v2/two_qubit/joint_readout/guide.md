# Four-branch joint readout

**Readers:** operator, agent. Run `two_qubit/joint_readout` and interpret its nine-class joint distribution.

## Before run

Bind `f1` and `f2` to different fluxonium components, `coupler` to their transmon, and `mux` to the MuxGroup. Open and Reset capture aliases, role names and original full-group tone positions. Refresh preserves authored inputs. Use Reset after changing sources or aliases.

The MuxGroup must contain ordered qubits and matching resonators for every tone. Keep the complete readout in that order, including tones outside F1/F2. The form shows the captured mapping as readonly fields.

Provide calibrated `$f1.module.x180` and `$f2.module.x180` pulses on different charge generators. Readout starts from `$mux.module.readout` or a blank mux input when the slot is absent. Confirm generator, mixer, ADC channels, integration window, trigger timing and headroom before Run.

F1 and F2 each require a circle classifier **with radius** in their readout tone. Run mux singleshot GE and apply GE Post under the actual readout conditions. Missing calibration, linear or centroid3 mode, and missing radius make the cfg invalid with a GE Post prompt. Other tones need no classifier for this experiment. Changing readout settings does not invalidate old calibration automatically. Recalibrate after changing them.

`reset` starts as None. Leave it None to omit reset, or supply a calibrated reset configuration to emit it at the start of every shot. Reset is separate from heralding. This version has no herald read. Set DC devices externally before Run.

## Sequence and acquisition

Each shot starts with optional reset, applies its nominal I/Xπ preparation, then reads the full mux once with tag `terminal`. One program sweeps two hardware axes, `prepared_f1` outside `prepared_f2`. Each axis uses 0 for g (I) and 1 for e (Xπ), giving row order gg, ge, eg, ee.

The program forks directly into F1 and F2 timelines after optional reset, without an extra delay_by_timeline. The forked timelines inherit the reset end. Each timeline has an aligned branch that emits its Xπ only for case 1; case 0 is empty. Both Xπ pulses start together for ee. The timelines join before terminal readout, so all preparations have the same readout start time. Without reset, preparations act on the thermal ensemble.

`reps` is shots per acquisition round per branch. `rounds` accumulates completed rounds at acquisition. Initial and relaxation delays use µs.

The processor classifies F1/F2 on each shot and sums the nine joint classes. Each role has g, e and other. Circle boundaries and equidistant points count as other. The experiment does not save per-shot IQ and does not multiply separate qubit probabilities.

The live plot shows terminal P(g,g) from accumulated counts for each nominal branch. Cancel preserves completed rounds. Unmeasured branches stay NaN. A failed acquisition remains an operation failure even when partial counts survive.

## Saved data and analysis

Result stores integer axes `prepared_f1=[0,1]` and `prepared_f2=[0,1]` (0=g, 1=e), and float64 `sums` with shape `(2,2,3,3)`. F1 is the outer preparation axis. The last axes are terminal F1 and F2, each ordered g/e/other. Counts remain unnormalized. Native persistence generates the class coordinates and saves the actual run cfg. File size does not grow with the shot count.

FIT shows a 4×9 count and row-normalized probability matrix. Columns are F1-major joint classes from g,g through other,other. Every class, including other, contributes to the denominator. Zero-count and unmeasured rows give NaN probabilities. Partial acquisitions can still provide distributions from their completed rounds. Each row label shows its acquired shot count.

The matrix includes initial populations, preparation errors and readout effects. It is not a pure-state assignment matrix. FIT offers no correction or writeback. Do not infer gate fidelity or invert this matrix without separate calibration evidence.

Old raw-IQ files lack `sums` and fail Load explicitly. Files with the old single preparation axis also fail Load. There is no conversion from saved IQ or old probabilities. Analyze uses only saved counts and needs no current library or cfg lookup.

## Heralded alternative

Open `two_qubit/joint_readout_heralded` for a separate experiment with its own cfg and saved Result. It adds a full mux herald read before preparation, saves all 81 herald/terminal class combinations, and aligns terminal distributions from herald g,g and e,e by actual preparation. It also adds their counts before normalizing a combined matrix. See the [heralded guide](guide_heralded.md).

Herald pass fraction measures classified acceptance, not ground-state purity. Neither version claims hardware calibration from offline checks. Continue physical calibration with the [FTF bring-up guide](../guide.md).

## Notebook

Use a validated `JointReadoutCfg` as `cfg`. Supply the notebook's existing
hardware handles and device manager. `point_view` is the bound PointView for the
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
