**Readers:** operator, agent. Search a bounded range of JPA settings for a measured candidate with high qubit readout separation.

# JPA joint search

Each candidate sets JPA DC bias, absolute pump RF frequency and pump source power together. It compares optional reset → no pulse → resonator readout (g) with optional reset → qubit π pulse → resonator readout (e). The search uses the score from one candidate to choose later settings. It leaves the devices at the last evaluated candidate, which may not be the best one.

## Set up the search

Choose the qubit, resonator and JPA aliases at the selected point, then choose **Defaults**.

- Select **JPA flux-bias source** and **JPA pump RF source**. Their initial values come from `$jpa.flux_dev` and `$jpa.rf_source`. Confirm output and safe bounds before Run. **JPA bias unit** starts at A; match it to the DC source. No conversion is performed.
- Set bounds for **JPA flux-bias value**, **JPA pump RF frequency**, and **JPA pump source power**. The initial ranges are `$jpa.flux` ± 0.005 in the bias unit, 0.98 to 1.02 times `$jpa.pump_freq` in MHz, and `$jpa.pump_power` ± 5 dBm. The three **Expts** values start at 101. Their ratios guide initial sample allocation; they are not a Cartesian scan or the total number measured.
- **Candidate budget** is the total number of evaluations, at least 4, initially 1001. **Optimizer seed** accepts a nonnegative integer. Equal seeds and measured score histories give equal choices; leave the seed empty for a new search.

Choose **Reps** for the shots per preparation and setting in each round, then **Rounds** for the number of such rounds. Defaults are 1000 × 10. The Analyze summary shows the Reps × Rounds from its source run. **Relaxation wait** starts at the qubit T1, or 30 µs if T1 is missing, and lets the system relax between shots. An invalid stored T1 leaves the wait blank; enter a calibrated delay.

Select **Qubit π pulse** from the qubit's X180 calibration and **Readout** from the resonator's readout calibration. Without a saved slot, Defaults uses the qubit RF frequency and drive channel for the π pulse, and the resonator frequency and readout channels for the resonator pulse. The readout trigger starts at the resonator flight time plus 0.015 µs. **Reset** is an optional pulse module before preparation. Leave it empty to rely on relaxation between shots.

## Choose a measured candidate

The separation SNR is the distance between the g and e IQ means divided by their pooled width along the line joining the means. **IQ shape penalty** defaults to zero. Increasing it downweights unequal widths or skew along that axis. The resulting g/e separation score (a.u.) is neither a state population nor an assignment fidelity.

The four panels show the same candidates by evaluation order, bias, pump RF frequency and source power. Color shows the search stage number, not a pulse phase or score. The order panel connects the scores in evaluation order to show whether better candidates appear late. The three setting panels show coverage and whether the best point lies at a bound; they are projections of a joint search, not independent sweeps. Live reports sampled candidates and retained shots. Analyze adds a red star for the same best joint candidate in each panel and labels all three settings and the score. There is no fit or shot-SEM bar; the score is not a per-shot average.

Recheck the complete selected tuple together, rather than mixing settings from different points. If the best repeatable setting is interior, narrow the bounds around it and repeat. Do not infer a global optimum from a boundary candidate or a run without stable separation. Subtract each previous setting from the selected one to obtain the adjustments in bias units, MHz and dB. Writeback offers `$jpa.flux`, `$jpa.pump_freq` and `$jpa.pump_power` separately; accepting them only updates the stored calibration.
