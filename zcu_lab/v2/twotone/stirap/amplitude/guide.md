**Readers:** operator, agent. Sweep the two STIRAP drive gains to find a region of high g-to-f transfer and low e label fraction.

## Sequence and key settings

Start Pump and Stokes with the chosen lead → wait for both pulse ends → one calibrated **Readout** → **Relax delay**. Run sweeps Pump gain outside Stokes gain in one round.

- Measure **Pump carrier 0–3** and **Stokes carrier 3–2** independently in MHz. On hardware, check that Pump and Stokes are wired to separate charge-drive lines so their pulses can overlap. Set signed **Pump gain** and **Stokes gain** ranges within [−1, 1] a.u.; neither is a calibrated rotation angle.
- In simulation, set **Pump generator** and **Stokes generator** to the `gen_ch`
  values of two drive lines for the qubit under test. The [Integer-fluxonium preset](../../../../simulation_envs/integer_fluxonium/field-guide.md#circuit-bath-and-routing)
  assigns Pump to 1 and Stokes to 2. In FTF, look up the generators in
  [`drive_lines`](../../../../simulation_envs/FTF/field-guide.md#drive-linescharge-與小訊號-flux-校準).
  Choose charge drive lines that couple to that qubit. Do not use generators
  in `flux_lines`, which play fast-flux pulses. The environment routes pulses
  by generator number and does not check the intended Pump or Stokes role.
- Choose each waveform and duration independently. An arbitrary waveform takes its duration from the asset. Positive **Stokes lead** starts Stokes first and delays Pump; negative reverses the order. Zero starts both together.
- Use a pulse readout with three saved g/e/f centers and a training confusion matrix from `singleshot/qutrit_population` at the same settings. Choose **Shots per point** for the needed precision.

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the maps and choose the gains

Run shows raw g/e/f label fractions. Pump gain is x and Stokes gain is y, both requested coordinates in a.u. A line through the origin is a fixed gain ratio. An incomplete round leaves blanks.

Analyze requires valid saved cfg and finite, nonnegative counts with a positive total at every grid point. A wrong classifier type still fails. With a usable saved matrix, it shows measured fractions above corrected fractions. If the classifier is absent, Analyze only shows **classified populations**. A singular matrix or a nonfinite solve at a valid point logs one warning and uses the same classified maps. The title's condition number and minimum singular value warn about noise amplification when respectively large and small. Corrected fractions are not clipped. Read the fraction from each panel’s own colorbar in Run and Analyze. These 2D maps do not show pointwise shot SEM. Preparation labels cannot establish pure state populations, transfer fidelity or the intermediate state.

The f cross marks the largest finite value on the displayed basis. Ties go to the first acquired point, and no finite value gives no peak. Without correction, the recipe reports `classified_f_peak` with the f label fraction and both gains. The corrected peak and its gains are null. Select a broad region of high f and low e, then refine the grid. If it reaches an edge, extend within the approved range before adopting the gains. No writeback is proposed.

For `stirap_amplitude`, null waveform or **Readout** selectors fail during selection, before overrides. Null channels, lead, shots or relax delay leaves a blank; required blanks after overrides fail readiness before Run. Missing centroid3 blocks acquisition. A successful Run can save even if Analyze fails.
