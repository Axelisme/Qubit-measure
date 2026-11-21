# Heralded two-qubit paired RB and CZ-interleaved RB

**Readers:** operator, agent. Compare gg- and ee-conditioned returns and decide whether a shared-decay fit constrains the CZ error.

Select `two_qubit/rb_heralded`. Use the calibrated point, bindings, pulses and Local Z pair in [Before run](guide.md#before-run). The Form fields and their initial values are the same as in ordinary RB except for Reset. Leave Reset blank to omit preparation, or select an ordinary reset. Heralded RB always performs its own herald read and does not accept a herald reset module. Open and Defaults reload the captured settings. Refresh keeps the draft and checks it against the new capture.

## Shot sequence

Each shot uses optional reset → joint herald readout → random Clifford prefix → recovery Clifford → joint terminal readout. Both reads use the same Mux readout, timing and classifiers. Both qubit tones need calibrated circle centers and radii from GE and GE Post. Repeat those calibrations after changing readout settings.

The recovery inverts the full sequence. For a shot classified as gg by the herald, the expected terminal result is gg. For ee herald, it is ee. The order is Qubit1, Qubit2 in both reads. These are classified return probabilities, not direct state preparation purity or gate fidelity. The classifier's other category does not identify a particular leakage state.

## Wait after herald

Set Mux readout's post-delay (µs) to let resonator photons decay before the Clifford sequence. The wait starts at the end of the readout waveform. Herald and terminal readout share the setting. Residual photons can shift the transition and reduce the effect of an Xπ pulse.

Start at `5 / (2πκ)` µs using the smallest readout linewidth κ in MHz from mux Onetone. For example, 1.2 MHz gives about 0.66 µs. To check the wait, use [`two_qubit/joint_readout`](../joint_readout/guide.md) with gg mux-herald reset. In the row with a nominal Xπ on the target, add the columns with terminal e on that qubit. Compare that sum at two waits. If it still rises at the longer wait, increase the post-delay before repeating RB. This sum is a classified e fraction among gg-accepted shots, not a gate fidelity.

## Sequences and cost

Each seed draws the reference sequence once. Both arms use its prefixes at the selected Random Clifford depths (count), and the interleaved arm inserts one CZ after each random Clifford. Each arm and depth gets its own recovery. See [Sequences and cost](guide.md#sequences-and-cost) for the pulse and frame order. Depth 0 checks the baseline without random Cliffords. Reps (count) sets shots at each depth per round; Rounds (count) sets rounds per seed and arm.

## Counts and complete seeds

Save keeps all joint herald and terminal outcomes for each seed, arm and depth, including ge, eg and other. If stopped, the run retains counts from finished rounds. Load uses the saved cfg and does not change the current Form. Incompatible older formats, including raw-IQ or a separate round axis, need the corresponding older release. Load does not convert or modify them.

In Analyze, the header identifies the experiment and source run. Reps × Rounds comes from the saved cfg, not the current Form. It does not report how many shots were taken or passed a herald condition.

## Joint decay and uncertainty

- **Conditional return**, the live and first Analyze figure, pools shots across seeds. The two panels show P(gg terminal | gg herald) and P(ee terminal | ee herald). Reference is blue and CZ-interleaved is orange. Each probability divides the number of matching terminal results by all shots with the indicated herald. Terminal other still counts in the denominator. The live caption gives measured seed/arm/depth cells and the total heralded shots across all outcomes, not the number in either fit group.
- **gg decay fit** and **ee decay fit** show separate equal-seed means and model curves for the two herald states. A fit uses at least four distinct depths and two seeds per herald with a defined return probability at all depths in both arms. The gg and ee groups can use different seeds. Each arm has one fitted decay `p` shared by both groups; amplitude and offset in `A p^depth + B` are separate. Check both groups before trusting the shared ratio.

Capped bars on the conditional-return figures show shot SEM. This is the sample standard deviation of the terminal success indicator divided by the square root of the matching herald shot count. A point with fewer than two matching shots has no SEM bar. A zero bar at probability 0 or 1 is not a confidence bound. Pale bars show one sample SD across the available seeds. The bars on the two fit plots show seed SD for the complete seeds in that group's fit. A single seed gives no seed SD bar. The fit uses its existing binomial variance weights, not the plotted seed SD or the pooled shot SEM.

The CZ error estimate is `3/4 × (1 − p_interleaved/p_reference)`. The reference error per Clifford is `3/4 × (1 − p_reference)`. Both are unitless. The 95% seed CI resamples whole identities from the union of valid seeds, keeping depths, arms and available herald groups together. The default is 1000 draws with bootstrap seed 42; at least 90% must give valid fits. The interval describes statistical seed variation and does not include readout error or coherent or gate-dependent noise. A negative estimate means the interleaved decay fit falls more slowly than reference. Retain it as a diagnostic, not a physical negative error rate.

The fitted 1/e depth and covariance standard error appear as `value ± error` in Clifford counts. Their ratio, error divided by depth, is the relative error. Both heralds have the same depth and error for each arm. R² measures how much of the point variation the model explains. Normalized RMS divides the root-mean-square residual by the observed range. The two heralds have separate residual metrics. No threshold selects a calibration. If the depth window does not resolve the decay, do not use the ratio as a CZ error, even if the fit returns a number. With a first nonzero depth of 1 Clifford and a fitted 1/e depth below 1, adding deeper points alone cannot capture that early drop. Check readout, pulse calibrations and the depth-0/1 contrast. If the sweep instead misses the tail of a resolved decay, add points near three times the fitted 1/e depth, such as depth 60 for a depth of 20. RB does not specify a direction or amount for changing CZ gain or frequency.

Layout reference: Ding, Fig. 5-13(b) (printed p. 157), separates RB data points and fitted curves. Here the dots represent classified returns conditioned on gg or ee herald, with a shared decay per arm.

## Herald fractions

Each arm has a separate **Herald outcome fractions** figure. The three panels fix Qubit1 to g, e or other. In each panel the legend identifies Qubit2's class. All nine probabilities use the same denominator, all heralded shots at that depth in that arm, pooled across seeds. Bars show shot SEM for each outcome. Use these panels to check whether one herald group has few shots or whether other outcomes dominate. For example, doubling Reps at fixed fractions reduces shot SEM by approximately `1/√2`; it does not improve a low gg fraction. If other dominates, check the saved circle calibrations before taking more data. The gg fraction measures classified acceptance, not purity.

## Notebook

Use a validated `RBHeraldedCfg` with the two saved tone positions and your existing SoC, device manager and `point_view`. Notebook does not infer the GUI bindings.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.rb.core_heralded import RBHeraldedExp, RBOptions

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(RBHeraldedExp())
record = exp.run(cfg)
analysis = exp.analyze(RBOptions())
saved = exp.save_run(point_view, source=record)
```
