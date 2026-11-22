**Readers:** operator, agent. Sweep Pump and Stokes offsets to refine the measured carriers for g-to-f transfer.

## Sequence and key settings

Start the two drives with **Fixed Stokes lead** → wait until both finish → calibrated **Readout** → **Relax delay**. Positive lead delays Pump; negative delays Stokes. Run steps Pump detuning within each Stokes detuning in one round.

- Measure **Pump carrier 0–3** and **Stokes carrier 3–2** separately; enter both in MHz. On hardware, use independent charge-drive lines. In simulation, select generators bound by the environment's wiring. Use signed gains in [−1, 1] and independent envelopes. Arbitrary envelopes take their duration from the asset.
- Set **Pump detuning** and **Stokes detuning** endpoints in MHz and point counts. Each offset adds to its own carrier. Cover the expected offsets without losing resolution near the carrier pair.
- Use a pulse readout with saved centroid3 centers and confusion matrix from `singleshot/qutrit_population` at the same settings. Circle **Other** is not f. **Relax delay** initially uses `5 × qubit.t1` (150 µs if absent).

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the maps and choose new carriers

Run shows measured g/e/f fractions, with requested Pump detuning on x and Stokes detuning on y. The title counts retained shots. Unfinished points remain blank. Analyze shows measured fractions above corrected fractions using the saved Run classifier. A large condition number or small minimum singular value means amplified count noise. Corrected fractions are not clipped. These 2D maps have no pointwise shot SEM and do not measure transfer fidelity or intermediate-state population.

The cross marks the largest finite corrected f sampled on the grid; ties use the first acquired point. Look for a stable high-f, low-e region. If it centers on Pump +0.5 MHz and Stokes −0.4 MHz, refine around those offsets, or add them to the respective carriers and center the next detuning scan at zero. No writeback is proposed.

For `stirap_detuning`, null waveform or **Readout** selectors fail during selection, before overrides. Null detuning endpoints/counts, carriers, channels, gains, lead, shots or relax delay leaves a blank; required blanks after overrides fail readiness before Run. Missing centroid3 blocks acquisition. A completed Run can still save if Analyze fails.
