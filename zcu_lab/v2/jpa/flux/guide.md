**Readers:** operator, agent. Sweep JPA DC bias and measure g/e readout separation at a fixed pump setting.

# JPA flux-bias sweep

Set pump RF, power and output before Run. At each bias, compare optional reset → readout (g) with optional reset → calibrated π pulse → readout (e).

## Set the sweep

- Bind the qubit, resonator and JPA. Verify **JPA flux-bias source** (initially `$jpa.flux_dev`). With it empty, Run records nominal bias coordinates without commanding a device. Match **JPA bias unit** to the connected source. **JPA flux-bias value** starts at `$jpa.flux` ± 0.005 in that unit, not Φ₀. Confirm the approved bounds before Run; values are not converted.
- Check **Qubit π pulse**, **Readout** and optional **Reset**. Fallback readout trigger starts at flight time + 0.015 µs. **Relaxation wait** starts at the captured T1 or 30 µs if missing. An invalid stored T1 leaves it blank; supply a calibrated wait.

For tab setup, refreshing sources, applying and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

## Read the score and choose a bias

The score divides g/e IQ-center separation by the pooled width along the separation axis. **IQ shape penalty** downweights unequal widths or skew. This is not assignment fidelity or state population.

Live counts sampled settings and shots. Live and Analyze connect the measured scores. Analyze overlays a dashed Gaussian smooth with a standard deviation of one scan step and marks the sampled bias at its peak. The smooth is not a fit. There is no shot SEM for this score. If the loaded cfg is missing, the axis shows native units rather than assuming A or V.

Use data taken with bias control. Check that the smooth peaks near a measured feature and away from the scan edge. Refine between its neighboring samples. The bias correction is selected minus previous bias in the source unit. Apply updates `$jpa.flux` without changing the device.
