**Readers:** operator, agent. Sweep the two STIRAP drive gains to find a region of high g-to-f transfer and low e label fraction.

## Sequence and key settings

Start Pump and Stokes with the chosen lead → wait for both pulse ends → one calibrated **Readout** → **Relax delay**. Run sweeps Pump gain outside Stokes gain in one round.

- Measure **Pump carrier 0–3** and **Stokes carrier 3–2** independently in MHz. Use separate charge-drive lines for overlap. Set signed **Pump gain** and **Stokes gain** ranges within [−1, 1] a.u.; neither is a calibrated rotation angle.
- Choose each waveform and duration independently. An arbitrary waveform takes its duration from the asset. Positive **Stokes lead** starts Stokes first and delays Pump; negative reverses the order. Zero starts both together.
- Use a pulse readout with three saved g/e/f centers and a training confusion matrix from `singleshot/qutrit_population` at the same settings. Choose **Shots per point** for the needed precision.

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the maps and choose the gains

Run shows raw g/e/f label fractions. Pump gain is x and Stokes gain is y, both requested coordinates in a.u. A line through the origin is a fixed gain ratio. An incomplete round leaves blanks.

Analyze shows measured fractions above corrected fractions, using the Run's saved calibration. The title's condition number and minimum singular value warn about noise amplification when respectively large and small. Corrected fractions are not clipped. These 2D maps do not show pointwise shot SEM. Preparation labels cannot establish pure state populations, transfer fidelity or the intermediate state.

The corrected f cross marks the largest finite sampled estimate; ties go to the first acquired point, and no finite value gives no peak. Select a broad region of high f and low e, then refine the grid. If it reaches an edge, extend within the approved range before adopting the gains. No writeback is proposed.

For `stirap_amplitude`, null waveform or **Readout** selectors fail during selection, before overrides. Null channels, lead, shots or relax delay leaves a blank; required blanks after overrides fail readiness before Run. Missing centroid3 blocks acquisition. A successful Run can save even if Analyze fails.
