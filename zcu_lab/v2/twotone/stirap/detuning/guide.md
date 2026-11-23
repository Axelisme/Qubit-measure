**Readers:** operator, agent. Sweep Pump and Stokes offsets to refine the measured carriers for g-to-f transfer.

## Sequence and key settings

Start the two drives with **Fixed Stokes lead** → wait until both finish → calibrated **Readout** → **Relax delay**. Positive lead delays Pump; negative delays Stokes. Run steps Pump detuning within each Stokes detuning in one round.

- Measure **Pump carrier 0–3** and **Stokes carrier 3–2** separately; enter both in MHz. On hardware, check that Pump and Stokes are wired to separate charge-drive lines. Use signed gains in [−1, 1] and independent envelopes. Arbitrary envelopes take their duration from the asset.
- In simulation, set **Pump generator** and **Stokes generator** to the `gen_ch`
  values of two drive lines for the qubit under test. The [Integer-fluxonium preset](../../../../simulation_envs/integer_fluxonium/field-guide.md#circuit-bath-and-routing)
  assigns Pump to 1 and Stokes to 2. In FTF, look up the generators in
  [`drive_lines`](../../../../simulation_envs/FTF/field-guide.md#drive-linescharge-與小訊號-flux-校準).
  Choose charge drive lines that couple to that qubit. Do not use generators
  in `flux_lines`, which play fast-flux pulses. The environment routes pulses
  by generator number and does not check the intended Pump or Stokes role.
- Set **Pump detuning** and **Stokes detuning** endpoints in MHz and point counts. Each offset adds to its own carrier. Cover the expected offsets without losing resolution near the carrier pair.
- Use a pulse readout with saved centroid3 centers and confusion matrix from `singleshot/qutrit_population` at the same settings. Circle **Other** is not f. **Relax delay** initially uses `5 × qubit.t1` (150 µs if absent).

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the maps and choose new carriers

Run shows measured g/e/f fractions, with requested Pump detuning on x and Stokes detuning on y. The title counts retained shots. Unfinished points remain blank. Analyze requires valid saved cfg and finite, nonempty counts across the grid. A wrong classifier type still fails. With a usable saved matrix, it shows measured fractions above corrected fractions. If the classifier is absent, Analyze only shows **classified populations**. A singular matrix or a nonfinite solve at a valid point logs one warning and uses the same classified maps. A large condition number or small minimum singular value means amplified count noise. Corrected fractions are not clipped. Read the fraction from each panel’s own colorbar in Run and Analyze. These 2D maps have no pointwise shot SEM and do not measure transfer fidelity or intermediate-state population.

The f cross marks the largest finite value on the displayed basis. Ties go to the first acquired point. Without correction, the recipe reports `classified_f_peak` with the f label fraction and both offsets from the carriers. The corrected peak and its offsets are null. Look for a stable high-f, low-e region. If it centers on Pump +0.5 MHz and Stokes −0.4 MHz, refine around those offsets, or add them to the respective carriers and center the next detuning scan at zero. No writeback is proposed.

For `stirap_detuning`, null waveform or **Readout** selectors fail during selection, before overrides. Null detuning endpoints/counts, carriers, channels, gains, lead, shots or relax delay leaves a blank; required blanks after overrides fail readiness before Run. Missing centroid3 blocks acquisition. A completed Run can still save if Analyze fails.
