**Readers:** operator, agent. Sweep JPA pump RF to find the sampled frequency that best separates g and e in readout IQ.

# JPA pump frequency

At each pump RF setting, compare optional reset → readout (g) with optional reset → calibrated π pulse → readout (e). The g/e SNR divides the IQ-center separation by the pooled width along the separation axis. **Skew penalty** weights the score down for distorted IQ clouds.

## Set the sweep

- Bind the qubit, resonator and JPA. Verify **JPA RF device** (initially `$jpa.rf_source`) and the approved frequency bounds. If left empty, Run records sweep coordinates without commanding a pump. **JPA pump freq** starts at 0.98–1.02 × `$jpa.pump_freq` in absolute MHz.
- Check **Pi pulse**, **Readout** and optional **Reset**. Use calibrated slots or fill the templates. The fallback readout trigger starts at flight time + 0.015 µs. **Relax delay** starts at the captured T1, or 30 µs if missing; an invalid stored T1 leaves it blank.
- **Scan-order seed** holds the shuffled order of the interior settings fixed across repeated runs. Leave it blank for a new order.

For tab setup, refreshing sources, applying and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

## Read the score and update calibration

Compare the sampled g/e SNR against pump RF in Live and Analyze. Analyze marks the sampled maximum; it is not a fit and has no shot SEM. Recheck an interior peak on a finer grid before using it. The correction is selected minus previous pump frequency in MHz. Apply updates `$jpa.pump_freq`; it does not retune the source.
