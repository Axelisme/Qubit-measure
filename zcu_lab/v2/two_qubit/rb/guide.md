# Two-qubit paired RB and CZ-interleaved RB

**Readers:** operator, agent. Compare reference and CZ-interleaved decays and decide whether the data constrain a CZ error estimate.

## Before run

Select `two_qubit/rb`. Bind `qubit1`, `qubit2`, `coupler` and `mux` to the calibrated point. Both qubits must appear in the full mux group. The tab uses their order in the mux group, not ADC channel numbers, to find the two readout tones. The outcome `(g,e)` always means Qubit1 in g and Qubit2 in e, regardless of ADC order.

| Form field | What to check |
| --- | --- |
| Qubit1 Xπ/2, Qubit2 Xπ/2, Qubit1 Xπ, Qubit2 Xπ | Open or Defaults copies the four pulses from the corresponding point slots. Use amplitude and length Rabi to calibrate each pulse. Check frequency (MHz), signed gain (a.u.) and envelope duration (µs). Each X pulse uses its qubit's CZ generator. |
| Two-line CZ drive | Defaults copies the two coupler pulses. Check their common frequency (MHz), envelope duration (µs) and start time, and keep the calibrated gain and constructive relative phase of each line. The interleaved arm inserts this pulse pair after each random Clifford. |
| Local Z corrections | Defaults reads the ordered pair from `$coupler.ext.cz_phase` and checks the component names. Use the corrections from [Local Z](../local_z/guide.md), usually minus the measured local-Z phase per CZ (deg). |
| Mux readout | Defaults selects `$mux.module.readout`. Use Lookback for the readout timing (µs), mux Onetone for readout frequencies (MHz) and gains (a.u.), and GE and GE Post for both circle centers and radii. Repeat GE and GE Post after changing readout settings. |
| Reset | Leave blank to start from the thermal population. Select ordinary reset to prepare each shot or `reset/mux_herald` to keep only shots classified as g on both qubits before the sequence. |
| Random Clifford depths (count) | Enter the number of random Cliffords before recovery, excluding the recovery itself. The default is 0, 1, 2, 3. Use depth 0 for the no-random-Clifford baseline. |
| Sequence seeds | Enter integer sequence IDs from 0 to 2147483647. The default is 0, 1. The two arms use the same sequence IDs; duplicates do not add independent sequences. |
| Reps (count), Rounds (count) | Enter shots per depth in each round and rounds per seed and arm. Defaults are 1000 and 10. Increase either to reduce shot noise without sampling more random sequences. |
| Initial delay (µs), Relax delay (µs) | Set the wait before the first shot of a round and the relaxation wait after each shot. The initial wait starts at 1 µs. Check the relaxation wait in the Form before running. |

After changing bindings, select Defaults to reload the calibrated values. Refresh keeps the draft and checks it against the new source. If an older point refers to previous role names, update the bindings and the `cz_qubit1` / `cz_qubit2` pulse slots, then rerun Local Z to get the ordered pair.

For a herald reset, select the same full-group library readout as Mux readout and set `keep` to g at both role tone positions. The module's optional `prepare` runs before the herald read. The readout delays apply to both reads. Ordinary RB saves terminal counts only for shots that pass; choose [heralded RB](guide_heralded.md) to retain all herald outcomes and compare gg and ee separately.

## Sequences and cost

Each shot uses optional reset → random Clifford prefix → recovery Clifford → joint terminal readout. The reference arm uses the random Cliffords alone. The interleaved arm adds one CZ after each random Clifford. Reference Cliffords can already contain CZs. Both arms use the same random prefix at each depth, drawn uniformly from the 11,520 two-qubit Cliffords. Each arm and depth gets a recovery that inverts its own sequence.

The two local frames start at zero. Virtual Z updates change the phase of later Xπ/2 and Xπ pulses. Each CZ uses the calibrated two-line phase and then applies the Local Z corrections to the frames. Both qubits finish each Clifford before starting the next. Depth and seed lists are sorted and deduplicated before acquisition.

## Save, cancel and analyze

Save retains joint g, e and other counts for each seed, arm and depth. If stopped, the run retains counts from finished acquisition rounds. Load restores those counts and the saved cfg without changing the current Form. Incompatible earlier formats, including raw-IQ files and files with a separate round axis, need the corresponding older release. Load does not convert or modify them.

The Analyze header identifies the experiment and the source run. Reps × Rounds comes from the saved run config, not the current Form, and does not state that every round finished or every shot passed herald.

## Interpreting the result

- **Joint ground return**, the live and first Analyze figure, shows the fraction of kept shots classified as terminal `(g,g)`, pooled across seeds by arm and depth. The live caption counts measured seed/arm/depth cells and total retained shots. Other outcomes contribute to the denominator. No kept shots means no plotted probability.
- **Paired decay fit** uses the same depths and outcome probability, but gives each seed equal weight. It uses only seeds with data at every depth in both arms, with at least four distinct depths and two complete pairs. Dots are means, solid curves are fits to `A p^depth + B`. Compare both curves with their dots before interpreting the ratio.

Capped bars on the pooled views are shot SEM, the sample standard deviation of the kept-shot success indicator divided by the square root of the kept shot count. Fewer than two kept shots leaves the SEM undefined. A zero bar at probability 0 or 1 is not a confidence bound. The pale bars on pooled views and the bars on the fit are the sample SD across seeds, not SEM or fit error. The fit figure uses complete paired seeds for those bars; pooled views use the seeds available at each point. Seed SD includes shot noise and variation between sequences. With fewer than two seeds, there is no seed SD bar.

The CZ error estimate is `3/4 × (1 − p_interleaved/p_reference)`. Reference error per Clifford is `3/4 × (1 − p_reference)`. Both are unitless. The 95% seed CI resamples whole paired seed curves. The default is 1000 resamples with bootstrap seed 42. The CI omits systematic effects such as readout errors and coherent or gate-dependent noise. A negative estimate means the fitted interleaved decay is slower than the reference decay. Keep the signed value as a model diagnostic, not a physical negative error rate. Other outcomes do not identify a particular leakage state.

Each arm's 1/e depth is in Clifford counts. Its ± value is the standard error from the fit covariance, not the shot SEM. Divide that error by the 1/e depth to obtain the reported relative error. R² measures how much of the spread in the fitted observations the model explains. Normalized RMS is the root-mean-square residual divided by the observed range. Neither is an acceptance threshold. A large relative error or a depth-window warning means the decay can be poorly constrained even with a large R². The caption names the affected arm and shows the fitted 1/e depth and the bound it reached or crossed, both in Clifford counts.

If the fitted 1/e depth is shorter than the first nonzero depth, add shallower depths. If the data only cover the start of a decay and the 1/e depth is inside the sweep, aim to include both the 1/e point and depths around three times it, then compare the fit uncertainty again. For example, a supported 1/e depth of 20 Cliffords with only shallow coverage calls for points around 60 Cliffords. Do not extrapolate a window-limited decay into a CZ gain or frequency correction. If the pooled shot SEM is small but the seed CI is broad, add independent seeds rather than only Reps. To investigate an unexplained ratio, inspect both arm fits and repeat the upstream CZ and Local Z calibrations before reporting a gate error.

Layout reference: Ding, Fig. 5-13(b) (printed p. 157), places the RB data beside their fitted decays. Here the vertical coordinate is the full-mux joint ground-return probability, and the error estimate uses the ratio of the two complete-pair decays.

## Notebook

Use a validated `RBCfg` with the two saved tone positions and your existing SoC, device manager and `point_view`. Notebook does not infer the GUI bindings.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.rb.core import RBExp, RBOptions

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(RBExp())
record = exp.run(cfg)
analysis = exp.analyze(RBOptions())
saved = exp.save_run(point_view, source=record)
```

## Calibration order and field units

Calibrate Xπ/2 and Xπ with Rabi and the mux timing with Lookback. Tune mux frequencies and gains with mux Onetone before GE and GE Post. Locate the two-line CZ with conditional spectroscopy and chevron, then measure Local Z. Check the matching pulse slots and corrections under [Before run](#before-run). The [expanded pulse and readout field reference](guide_heralded.md#expanded-pulse-and-readout-fields) applies to both RB Forms.
