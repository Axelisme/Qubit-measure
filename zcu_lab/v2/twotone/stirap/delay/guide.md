**Readers:** operator, agent. Sweep the relative start time of Pump and Stokes to tune g-to-f transfer at fixed carriers and gains.

## Sequence and key settings

Start Pump and Stokes with the swept **Stokes lead** → wait for the later pulse end → calibrated **Readout** → **Relax delay**. Lead is the offset between pulse starts, not centers or ends. At +0.10 µs Pump starts 0.10 µs after Stokes; negative starts Pump first; zero starts both together.

- Enter measured **Pump carrier 0–3** and **Stokes carrier 3–2** in MHz. On hardware, check that Pump and Stokes are wired to separate charge-drive lines. Use signed gains in [−1, 1] and independent envelopes. For arbitrary envelopes, duration comes from the assets. Choose lead endpoints that cover the expected overlap.
- In simulation, set **Pump generator** and **Stokes generator** to the `gen_ch`
  values of two drive lines for the qubit under test. The [Integer-fluxonium preset](../../../../simulation_envs/integer_fluxonium/field-guide.md#circuit-bath-and-routing)
  assigns Pump to 1 and Stokes to 2. In FTF, look up the generators in
  [`drive_lines`](../../../../simulation_envs/FTF/field-guide.md#drive-linescharge-與小訊號-flux-校準).
  Choose charge drive lines that couple to that qubit. Do not use generators
  in `flux_lines`, which play fast-flux pulses. The environment routes pulses
  by generator number and does not check the intended Pump or Stokes role.
- Select a calibrated pulse readout with saved centroid3 centers and a 3×3 training confusion matrix from `singleshot/qutrit_population`. This scan does not train a classifier. **Shots per point** sets the sample count in the single acquisition round.

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the plots

Live shows measured g/e/f fractions versus requested Stokes lead in µs; unfinished points stay blank. Analyze skips unfinished or empty points and rejects a scan with no completed points. It requires valid saved cfg and rejects a wrong classifier type. With a usable saved matrix, it plots measured and corrected fractions in separate panels. Corrected values can lie outside [0, 1].

If the saved cfg lacks a classifier, Analyze only shows **classified populations**. A singular matrix or a nonfinite solve at a valid point logs one warning and uses that same view.

One bar is one shot SEM. For measured fraction p from N > 1 retained shots, SEM is √[p(1−p)/(N−1)]. The corrected panel propagates the retained shots' three-label count covariance with the saved calibration matrix treated as known. The bars do not show fit errors or shot or round spread. Missing, empty or single-shot points have no bar.

A large matrix condition number or small minimum singular value means the correction amplifies count noise. Preparation-label fractions are not transfer fidelity or proof of state purity.

## Next step

The dotted marker is the largest finite sampled f value on the displayed basis, not a global optimum. Ties use the first acquired point; no finite point yields no peak. Without correction, the recipe reports this point as `classified_f_peak` with its f label fraction and Stokes lead. It leaves the corrected peak and its lead null. If the marker is at the +0.10 µs endpoint, extend that endpoint while retaining the original overlap, then repeat. No writeback is proposed.

For `stirap_delay`, null waveform or **Readout** selectors fail during selection, before overrides. Null channels, shots or relax delay leaves a blank; required blanks after overrides fail readiness before Run. Missing centroid3 blocks acquisition. A completed Run can still save if Analyze fails.
