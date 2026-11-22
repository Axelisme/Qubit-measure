**Readers:** operator, agent. Sweep JPA pump source power to find the sampled power that best separates g and e in readout IQ.

# JPA pump power

At each pump power, compare optional reset → readout (g) with optional reset → calibrated π pulse → readout (e). The g/e SNR divides the IQ-center separation by the pooled width along the separation axis. **Skew penalty** weights the score down for distorted IQ clouds.

## Set the sweep

- Bind the qubit, resonator and JPA. Confirm the required **JPA RF device** (initially `$jpa.rf_source`) and approved limits. **JPA pump power** starts at `$jpa.pump_power` ± 5 dBm; check that both endpoints are allowed.
- Check **Pi pulse**, **Readout** and optional **Reset**. Use calibrated slots or fill the templates. The fallback readout trigger starts at flight time + 0.015 µs. **Relax delay** starts at the captured T1 or 30 µs if missing; an invalid stored T1 leaves it blank.
- **Scan-order seed** holds the shuffled order of the interior powers fixed across repeated runs. Leave it blank for a new order.

For tab setup, refreshing sources, applying and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

## Read the score and update calibration

Compare sampled g/e SNR against pump source power in dBm. Analyze marks the sampled maximum. There is no fit or shot SEM. Recheck an interior peak on a finer grid. Selected minus previous pump power gives the adjustment in dB. Apply updates `$jpa.pump_power` without commanding a power change.
