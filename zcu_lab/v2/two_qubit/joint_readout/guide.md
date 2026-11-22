**Readers:** operator, agent. Compare the four nominal two-qubit preparations with their terminal readout outcomes.

# Four-branch joint readout

Each shot runs **optional Reset → I or Xπ on each qubit → terminal mux readout**. The two Xπ timelines start together in the e,e branch; all four preparations have the same wait from preparation start to readout. Without reset, I and Xπ act on the thermal ensemble. The matrix measures preparation plus readout, not pure-state assignment accuracy.

## Set up the form

- Bind distinct `qubit1` and `qubit2` and their `mux`; no coupler is needed. Check [tone order](../guide.md#綁定位置與讀出-tones). Plots put Qubit1 first regardless of ADC or tone order.
- **Qubit1 Xπ** and **Qubit2 Xπ** start as editable Inline copies from the respective `x180` slots, or blank for a missing slot. Use the Rabi-calibrated pulse on each qubit's own generator. Editing an Inline copy does not change the library. A named pulse in the menu is not necessarily calibrated for that qubit.
- When filling an Inline pulse, keep its calibrated **Gen ch**, **NQZ**, RF **Frequency**, signed **Gain**, **Phase** and **Waveform → Length** together. **Pre-delay** and **Post-delay** are outside the envelope. **Mixer freq** can stay blank for an unmixed drive.
- Use the complete **Mux readout** from its slot (or fill the blank template). Match the readout frequency, gain and integration window used for the two tones' GE circles and GE Post radii. Blank **Reset** means no reset. For mux herald, use the [same readout and keep keys](../guide.md#herald-reset-與獨立-heralded-入口).
- Set **Relaxation wait** from qubit relaxation and cavity-emptying measurements, not from the 1 µs form initial. **Reps × Rounds** gives attempted shots per preparation; multiply by four for the run.

## Read the plots

Every retained shot counts once. The nearer circle center wins only if it is strictly nearer and the point is inside its radius; ties and points on or outside the radius are `other`. Other stays in the denominator and does not identify a leakage level.

- **Run and Analyze distributions:** one panel per nominal preparation, nine terminal classes on the x-axis, probability on the y-axis. Hatching marks the outcome matching that panel's preparation. Live adds the total retained shots across all four panels. Analyze shows the same distribution at the end of the run. No retained shots means no probabilities.
- **Error bars:** the sample standard deviation of the 0/1 class indicator divided by √N retained shots in that panel. This shot SEM is not single-shot spread, fit error, round spread or a confidence interval. Fewer than two shots gives no bar; a zero bar does not establish an impossible or certain event.
- **Analyze matrix:** rows are nominal preparations and columns are terminal classes; the upper number is probability, the lower number is count, and 0–1 color indicates probability. For exact fractions, divide the count by its row total. Use the source Run ID and **Reps×Rounds** in Analyze; the count alone does not identify the Reset/filter settings. Use that run's saved cfg, not the current form.

## Decide what to check next

Compare matching and mismatched outcomes within each row, and compare the e,e row with the single-flip rows. For one qubit's marginal, sum all columns with that qubit in the selected terminal class. If counting uncertainty dominates, collecting four times as many shots at fixed fractions and retention halves nonzero shot SEM. Large well-resolved mismatches call for checking GE separation and Other, then comparing herald or rechecking Xπ, not more shots alone. Do not invert this matrix or infer gate fidelity. There is no fit or writeback.

For legacy files see the [family load note](../guide.md#opendefaults-與人工-apply). For common GUI actions, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
