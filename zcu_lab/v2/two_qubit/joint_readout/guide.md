# Four-branch joint readout

**Readers:** operator, agent. Run `two_qubit/joint_readout` and interpret its nine-class joint distribution.

## Before run

Bind `f1` and `f2` to different fluxonium components, `coupler` to their transmon, and `mux` to the MuxGroup. Open and Reset capture aliases, role names and original full-group tone positions. Refresh preserves authored inputs. Use Reset after changing sources or aliases.

The MuxGroup must contain ordered qubits and matching resonators for every tone. Keep the complete readout in that order, including tones outside F1/F2. The form shows the captured mapping as readonly fields.

Provide calibrated `$f1.module.x180` and `$f2.module.x180` pulses on different charge generators. Readout starts from `$mux.module.readout` or a blank mux input when the slot is absent. Confirm generator, mixer, ADC channels, integration window, trigger timing and headroom before Run.

F1 and F2 each require a circle classifier **with radius** in their readout tone. Run mux singleshot GE and apply GE Post under the actual readout conditions. Missing calibration, linear or centroid3 mode, and missing radius make the cfg invalid with a GE Post prompt. Other tones need no classifier for this experiment. Changing readout settings does not invalidate old calibration automatically. Recalibrate after changing them.

`reset` starts as None. Leave it None to omit reset, or supply a calibrated reset configuration to emit it at the start of every shot. Reset is separate from heralding. This version has no herald read. Set DC devices externally before Run.

## Sequence and acquisition

Each branch starts with optional reset, applies its nominal I/Xπ preparation, then reads the full mux once with tag `terminal`. Branch order is 00, 01, 10, 11, with F1 first. Branch 11 applies F1 Xπ then F2 Xπ. Without reset, preparations act on the thermal ensemble.

`reps` is shots per acquisition round per branch. `rounds` accumulates completed rounds at acquisition. Initial and relaxation delays use µs.

The processor classifies F1/F2 on each shot and sums the nine joint classes. Each role has g, e and other. Circle boundaries and equidistant points count as other. The experiment does not save per-shot IQ and does not multiply separate qubit probabilities.

The live plot shows terminal P(g,g) from accumulated counts for each nominal branch. Cancel preserves completed rounds. Unmeasured branches stay NaN. A failed acquisition remains an operation failure even when partial counts survive.

## Saved data and analysis

Result stores `prepared_states=[0,1,2,3]` for named states gg, ge, eg, ee and float64 `sums` with shape `(4,3,3)`. The last axes are terminal F1 and F2, each ordered g/e/other. Counts remain unnormalized. Native persistence generates the class coordinates and saves the actual run cfg. File size does not grow with the shot count.

FIT shows a 4×9 count and row-normalized probability matrix. Columns are F1-major joint classes from g,g through other,other. Every class, including other, contributes to the denominator. Zero-count and unmeasured rows give NaN probabilities. Partial acquisitions can still provide distributions from their completed rounds. Each row label shows its acquired shot count.

The matrix includes initial populations, preparation errors and readout effects. It is not a pure-state assignment matrix. FIT offers no correction or writeback. Do not infer gate fidelity or invert this matrix without separate calibration evidence.

Old raw-IQ files lack `sums` and fail Load explicitly. There is no conversion from saved IQ or old probabilities. Analyze uses only saved counts and needs no current library or cfg lookup.

## Heralded alternative

Open `two_qubit/joint_readout_heralded` for a separate experiment with its own cfg and saved Result. It adds a full mux herald read before preparation, saves all 81 herald/terminal class combinations, and aligns terminal distributions from herald g,g and e,e by actual preparation. It also adds their counts before normalizing a combined matrix. See the [heralded guide](guide_heralded.md).

Herald pass fraction measures classified acceptance, not ground-state purity. Neither version claims hardware calibration from offline checks. Continue physical calibration with the [FTF bring-up guide](../guide.md).
