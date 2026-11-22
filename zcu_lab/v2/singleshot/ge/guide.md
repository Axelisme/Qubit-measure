**Readers:** operator, agent. Calibrate single-shot ground/excited readout and interpret Primary and Post.

# Single-shot GE

## Run

Repeat `optional Reset → optional initial preparation pulse → π probe off or on → Readout → Relax delay`. The two preparations differ only by the calibrated g↔e π probe pulse. Each produces a cloud of integrated single-shot complex IQ. The I and Q values are in normalized ADC units (a.u.).

| Run field | What to enter or check |
| --- | --- |
| Reset | Select the reset module before both preparations, or leave it empty. Reset is a pulse-sequence choice; Defaults reloads form values. |
| Initial preparation pulse | Select a pulse before the probe, or leave it empty to start from the relaxed state. If the pulse prepares e, select excited for **State before π probe**. |
| π probe pulse | Select the g↔e π pulse calibrated by Rabi on this qubit. The probe-on branch applies it once. |
| Readout | Select the resonator readout used for both clouds. Check RF frequency, gain and integration window. Readout duration and alignment affect cloud separation and relaxation during the window. A starting point for the post-readout cavity wait is 5/(2π·κ). |
| Shots per preparation (count) | Number of IQ shots **in each** branch, not divided between them. Open/Defaults starts at 100,000 each. |
| Relax delay (µs) | Wait after each repetition so the starting state can recover. Defaults uses five times the captured qubit T1. Missing T1 gives 150 µs; invalid stored T1 leaves this field blank. |

Open or Defaults captures the bound qubit, its resonator and their module choices. The π probe comes from the qubit's x180 slot, and Readout comes from the resonator's readout slot. If a slot is missing, the corresponding template supplies starting values from the captured frequency and wiring. Check unfilled timing and drive gain before Run. Refresh keeps the current draft; use Defaults to capture new calibration sources. For a readout template, leave 0.015 µs before and after the integration window within the RF pulse, starting from the measured time of flight.

The live IQ scatter uses blue for π probe off and orange for π probe on. The shot counts show which branch has arrived. The same colors, I and Q axes, and markers appear in Primary. Neither branch is guaranteed to be a pure energy state.

## Primary

| Primary field | What it does |
| --- | --- |
| State before π probe | ground maps off/on to g/e preparation; excited swaps that mapping. This changes interpretation, not acquisition. |
| IQ rotation method | pca uses the pooled axis of largest IQ spread; center uses the direction between the two centers. Both rotate IQ to separate the clouds along I. |
| Log scale for projected shot counts | Use a logarithmic count axis to inspect sparse tails. This changes only the measured histogram display, not the fit or IQ scatter. |
| Share decay ratio across histogram fits | Fit the two preparation histograms with a shared readout-length to relaxation-time ratio when enabled. |
| Readout length / T1 (a.u.) | Enter the known ratio of integration time to T1, or leave blank to estimate it from the data. Example: 1 µs / 20 µs = 0.05. |

Primary adds the dashed decision boundary where the real part of IQ rotated by the shown angle equals **Rotated I threshold**. On either side, it assigns the dominant preparation label. **Prepared-label assignment fidelity** is the average of the correct-label fractions across the two preparations (equal weight). It includes preparation mixing, so it is not pure-state readout fidelity. In the measured IQ projection, compare the shot-count distributions on the rotated I axis. Substantial overlap at the dashed threshold means more ambiguous assignments. The steps show measured counts, not fitted curves.

Use the Analyze summary's run ID and **Reps×Rounds** to identify the data. For example, 4096×1 per preparation means 4096 configured repeats in each branch, one round. An interrupted run can contain fewer retained shots; check the live counts.

The fit plot reports R² and normalized RMS for the joint fit and for each preparation histogram (ground and excited). R² measures the fraction of observed variance explained by the model. Normalized RMS is the residual RMS divided by the observed count range for that fit diagnostic. The two preparation values help locate a poor fit; they have no separate thresholds. Relative width error is the uncertainty of the fitted width divided by the fitted width. Primary leaves the proposed calibration unchecked when joint R² is below 0.9, joint normalized RMS exceeds 0.1, relative width error exceeds 0.5, or one of these checks is unavailable. The plot gives the failed check; a high empirical assignment score does not override it.

## Post and next steps

Post takes the Primary centers and populations without running another fit. The two scatter panels show classification around the g and e centers: blue for g, red for e, green for other. Dashed circles show the radius. In an overlap the nearer center wins; outside both circles the result is other. Other is not an identified leakage level. Post does not apply the Primary threshold.

Compare the two matrices on a 0–1 scale. **Fitted preparation populations** has preparation rows and model-state columns. **Corrected confusion** has model-state rows and assigned-outcome columns. It uses the fitted preparation mixing to undo preparation contamination. Both matrices label the `other` row as model only: the model isolates that row as `0%, 0%, 100%`. No third preparation was measured. This assumption does not replace the `other` column, which gives the model fraction or assigned outcome for each row.

Diagonal entries near one support discrimination only if both the histogram fit and the correction are credible. Post also leaves the proposal unchecked if the condition number of the **corrected confusion matrix** exceeds 10 or is unavailable. A large value means that errors in the assigned fractions can grow when you use that matrix to infer populations. It does not describe the earlier solve that removes preparation mixing.

If cloud overlap is high, do not write back the calibration just because the fit returned a threshold. Check the π calibration first if the off/on clouds have similar weights. If the prepared clouds are distinct but broad, vary the readout RF frequency, gain or integration window one at a time and repeat GE with the same preparation and shot count. Prefer the setting that improves the equal-weight assignment score without degrading joint fit quality or increasing Other. The IQ plots alone do not give a signed frequency correction. After comparing settings, apply the selected centers in Primary and radius/confusion in Post to the chosen Readout module. Neither action changes the qubit π pulse.

There is no per-point shot SEM in these IQ scatters or matrix views. A point is a single shot, not a swept mean. The recipe `singleshot_ge` uses the same fields and asks before applying either writeback.
