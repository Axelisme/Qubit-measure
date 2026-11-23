**Readers:** operator, agent. Search bias, pump RF and power together for a measured setting with high g/e readout separation.

# JPA joint search

At each candidate, set all three JPA device values. Compare optional reset → readout (g) with optional reset → calibrated π pulse → readout (e). The score is g/e IQ-center separation divided by the pooled width along the separation axis. **IQ shape penalty** downweights unequal widths or skew. The score is not state population or assignment fidelity.

## Set the search

- Bind the qubit, resonator and JPA. Check the calibrated **Qubit π pulse**, **Readout** and optional **Reset**. The fallback readout trigger starts at flight time + 0.015 µs. **Relaxation wait** starts at the captured T1, or 30 µs if missing; enter a calibrated wait if an invalid T1 leaves it blank.
- Confirm **JPA flux-bias source** and **JPA pump RF source**, initially the JPA's stored device names. Match **JPA bias unit** to the source; there is no conversion. Review both devices and the approved bounds before Run.
- **JPA flux-bias value**, **JPA pump RF frequency**, and **JPA pump source power** start at stored bias ± 0.005 source units, 0.98–1.02 × stored pump frequency, and stored power ± 5 dBm. The ratios of the three **Expts** counts allocate initial samples; they do not define a Cartesian grid. **Candidate budget** sets the total, at least 4. Leave **Optimizer seed** empty for a new search; the same seed and score history give the same choices.

For tab setup, refreshing sources, applying and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

## Read the search and choose a setting

The four panels show the same points by evaluation order, bias, RF and power. Color is search stage. The order panel connects scores to show whether they improve late. The three setting panels are scatter views of the same joint search, not independent sweeps; check if the best point is at a bound. Analyze marks the same best joint setting in all four panels with a red star. There is no fit or shot SEM.

The devices remain at the **last evaluated** setting, not necessarily the best one. Verify the best tuple together; do not combine axis values from different points. Repeat with narrower bounds around a stable interior best candidate. Apply stores the selected `$jpa.flux`, `$jpa.pump_freq` and `$jpa.pump_power` proposals, without commanding the devices.
