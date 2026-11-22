**Readers:** operator, agent. Sweep the JPA DC bias and choose a measured setting using qubit readout separation.

# JPA flux-bias sweep

Fix the pump RF frequency, power and output before Run. Each bias setting compares optional reset → no pulse → resonator readout (g) with optional reset → qubit π pulse → resonator readout (e).

## Set up the scan

Choose the qubit, resonator and JPA aliases, then choose **Defaults**. **JPA flux-bias source** starts from `$jpa.flux_dev`. If it is empty, the scan records nominal coordinates without changing the bias. **JPA bias unit** starts at A. Match the connected source to A, V or its native units. Sweep values are sent unchanged. **JPA flux-bias value** starts at `$jpa.flux` ± 0.005 in the chosen source unit, with 101 samples. Review the bounds and unit against the connected device before Run. These are source settings, not flux in units of Φ₀.

Choose **Reps** for the shots per preparation and setting in each round, then **Rounds** for the number of such rounds. Defaults are 1000 × 10. The Analyze summary shows the Reps × Rounds from its source run. **Relaxation wait** starts at the qubit T1, or 30 µs if T1 is missing, and lets the system relax between shots. An invalid stored T1 leaves the wait blank; enter a calibrated delay.

Select **Qubit π pulse** from the qubit's X180 calibration and **Readout** from the resonator's readout calibration. Without a saved slot, Defaults uses the qubit RF frequency and drive channel for the π pulse, and the resonator frequency and readout channels for the resonator pulse. The readout trigger starts at the resonator flight time plus 0.015 µs. **Reset** is an optional pulse module before preparation. Leave it empty to rely on relaxation between shots.

## Choose a bias

The separation SNR is the distance between the g and e IQ means divided by their pooled width along the line joining the means. **IQ shape penalty** defaults to zero. Increasing it downweights unequal widths or skew along that axis. The resulting g/e separation score (a.u.) is neither a state population nor an assignment fidelity.

Live counts sampled settings and shots and displays the measured scores. Analyze retains the same dots, overlays a Gaussian smooth with a standard deviation of one scan step, and marks the sampled bias where that smooth peaks. The solid line is a smooth, not a fit. There are no pointwise shot-SEM bars; an SEM of mean IQ is not uncertainty in the score. If a loaded run lacks its cfg, the axis uses native units without assuming A or V.

Use a complete scan taken with bias control. Check that the smoothed peak is near a feature in the measured dots and away from a scan edge. Repeat between its adjacent bias samples before accepting it. Subtract the old bias from the selected bias in the source unit to find the adjustment; a positive value increases the setting. Accepting the writeback updates `$jpa.flux` without changing the source value.
