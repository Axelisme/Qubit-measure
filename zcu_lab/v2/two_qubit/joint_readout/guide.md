**Readers:** operator, agent. Set up four-branch joint readout and compare the observed outcomes with the nominal preparations.

# Four-branch joint readout

Joint readout checks whether the two qubits return the expected pair of classifications after preparation. Each shot runs **optional reset → I or Xπ on each qubit → one terminal mux readout**. A nominal g means no pulse (I). A nominal e means apply Xπ. The two Xπ pulses start together in the e,e branch. The four branches wait the same time from the start of preparation to readout. Without Reset, the pulses act on the initial thermal ensemble, so g and e name the intended preparation, not a verified initial state.

## Set up the form

1. Bind the point's `qubit1` and `qubit2` aliases to different components and `mux` to their MuxGroup. This experiment does not need `coupler`. In the [MuxGroup](../guide.md#綁定位置與讀出-tones), find each qubit's position in the ordered qubit list. The same position in the resonator and readout tone lists must belong to that qubit. For example, if the group's qubits are ordered Q2, Q1, the first tone belongs to Q2. The plots always show Qubit1 first, independent of this order.
2. **Qubit1 Xπ** and **Qubit2 Xπ** show **Inline** when the form has copied the point's Xπ calibration from `$qubit1.module.x180` and `$qubit2.module.x180`. Inline means the pulse values are editable in this form. Editing a copy does not change the library. A missing calibration slot leaves the pulse field blank. A named choice refers to a library pulse; its presence in the list does not make it a calibrated Xπ for the selected qubit. Use each qubit's amplitude or length Rabi calibration.

   **Gen ch** is the qubit drive generator channel. Check that the two pulses use different channels. The RF frequency (MHz) sets the pulse carrier; signed gain (a.u.) scales the drive; waveform and length (µs) set the envelope shape and duration. Use the shape, length and gain from the same calibration. **NQZ** selects the DAC Nyquist band, phase (°) sets the rotation axis, and the before/after waits (µs) set spacing around the envelope. Leave mixer frequency blank if the drive does not use a mixer.
3. **Mux readout** starts at `$mux.module.readout` or a blank mux form when that slot is missing. Keep every tone in the group's order, even tones other than the selected pair. Use [mux single-shot GE and GE Post](../../mux/singleshot_ge/guide.md) to obtain g/e circle centers and radii for the selected pair. Recheck the calibration if you change the readout frequency, gain or integration window. The form checks the number of tones; it does not infer the order from ADC numbers or frequencies. Refresh keeps the draft and rechecks it against current bindings. Defaults replaces the draft with values from current sources.
4. **Reset** starts blank, so the shot starts without a reset or a herald filter. Choose a calibrated preparation reset to apply it in all four branches. For a herald filter, choose `reset/mux_herald`, use the same full mux readout, and set `keep` to g for both qubits. Set the keys to the zero-based positions in the full mux readout, not ADC numbers. Use `{'0': g, '1': g}` only for the pair at positions 0 and 1. The count then includes only shots that passed herald. The pass fraction is retained shots divided by attempted shots, not the true ground-state purity.
5. **Reps** is attempted shots per preparation per round. **Rounds** repeats the four preparations and adds their counts. The form starts at 1000 × 10, so a complete unfiltered run attempts 10,000 shots for each preparation and 40,000 across all four. **Initial wait** is before the first shot of a round. **Relaxation wait** is after a shot, before the next preparation. Both start at 1 µs. Set the relaxation wait from the device's relaxation and cavity-emptying calibration rather than treating 1 µs as a physical recommendation.

## Read the plots

For each retained shot, the mux classifier assigns g, e or other independently to each qubit, then the experiment counts the resulting pair. A point is g or e only if it is strictly closer to that center and inside its circle. Ties and points at or beyond the nearer circle's radius count as `other`. This class does not identify a leakage level. Every class, including `other`, contributes to its preparation's total.

- **Run and Analyze distributions.** Each panel fixes one nominal preparation. The x-axis lists the nine terminal classes in Qubit1, Qubit2 order. The y-axis is the fraction of that preparation's retained shots assigned to the class. The bars all use the same gray. Hatching marks the terminal class that matches the panel's nominal preparation. The same probability scale is used in each panel. Liveplot adds the total retained shots across the four panels, not the number per panel. The final liveplot and first Analyze figure otherwise use the same counts and display. A panel with no retained shots has no probabilities.
- **Error bars.** Each bar uses shot SEM, the sample standard deviation of the 0-or-1 class indicator divided by the square root of the retained shots in that panel. It is an estimate of uncertainty in the probability, not single-shot spread, fit uncertainty or a confidence interval. No bar is drawn for fewer than two retained shots. A measured fraction of exactly zero or one has zero sample SEM, which does not prove the event impossible or certain.
- **Analyze probability and count matrix.** Read rows as nominal preparations and columns as terminal classes. The upper number in each cell is its probability, rounded to three decimals; the lower number is the shot count. The colorbar runs from 0 to 1 on the same probability scale for all rows. Use the matrix to compare one outcome between preparations or recover numbers from small bars: divide its count by the sum of the nine counts in that row. A row without retained shots has no probability.

## Decide what to check next

For each row, check whether the matching outcome dominates the other outcomes. Quantify ambiguous runs by comparing the matching and largest mismatched fractions. If Qubit1 stays g on many shots after its nominal Xπ, sum the Qubit1=g columns in the e,g row, then compare the same quantity in the g,g row. This is a terminal Qubit1 marginal, not a gate error rate. Compare the e,e row to both single-flip rows to look for errors that appear only when both pulses play.

If the matching and mismatched bars are close within the shot uncertainty, increase the attempts without changing the preparation or readout. For example, to halve a nonzero SEM at the same retention and outcome fractions, collect about four times as many shots per preparation. If the bars already separate beyond their shot uncertainty but the mismatched fractions are large, more shots refine the probabilities; they do not fix preparation or readout. Check GE separation and `other` first. If the readout is stable, compare runs with and without g,g herald at fixed pulse settings, or recalibrate one Xπ at a time. The matrix alone cannot tell you which knob to change, by how much, or what the gate fidelity is. It mixes initial populations, preparation and readout; it is not a calibrated assignment matrix to invert.

Analyze identifies the source run and displays **Reps×Rounds** from its saved cfg. Use the matrix counts to find how many shots each preparation retained. Counts alone do not identify whether you used Reset or which shots a filter rejected. For that distinction, check the saved settings for the Analyze Run ID. The current form, including its Reset default, is not a record of a previous run. If a run is stopped, completed rounds still contribute. Analyze reads the saved counts without consulting the current form or library and does not propose writeback. The native file stores the cfg and a 4 × 9 set of terminal counts across two preparation and two classification axes, not individual shots. To open data from the discontinued standalone heralded experiment or earlier incompatible axis layouts, use the software version that wrote the file.

Continue in the [two-qubit calibration guide](../guide.md).

## Notebook

With a validated `JointReadoutCfg`, set `qubit1_tone` and `qubit2_tone` to the two positions in the full mux readout. Pass the existing hardware handles and device manager to the notebook adapter. Use a destination `ParameterView` for `point_view`.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.joint_readout.core import JointReadoutExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(JointReadoutExp())
record = exp.run(cfg)
analysis = exp.analyze(None)
saved = exp.save_run(point_view, source=record)
```
