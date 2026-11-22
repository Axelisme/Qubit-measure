**Readers:** operator, agent. Sweep the relative start time of Pump and Stokes to tune g-to-f transfer at fixed carriers and gains.

## Sequence and key settings

Start Pump and Stokes with the swept **Stokes lead** → wait for the later pulse end → calibrated **Readout** → **Relax delay**. Lead is the offset between pulse starts, not centers or ends. At +0.10 µs Pump starts 0.10 µs after Stokes; negative starts Pump first; zero starts both together.

- Enter measured **Pump carrier 0–3** and **Stokes carrier 3–2** in MHz. Choose separate charge-drive lines, signed gains in [−1, 1] and independent envelopes. For arbitrary envelopes, duration comes from the assets. Choose lead endpoints that cover the expected overlap.
- Select a calibrated pulse readout with saved centroid3 centers and a 3×3 training confusion matrix from `singleshot/qutrit_population`. This scan does not train a classifier. **Shots per point** sets the sample count in the single acquisition round.

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the plots

Live shows measured g/e/f fractions versus requested Stokes lead in µs; unfinished points stay blank. Analyze skips unfinished or empty points and plots measured and corrected fractions in separate panels, using the Run's saved calibration. The corrected values can lie outside [0, 1].

One bar is one shot SEM. For measured fraction p from N > 1 retained shots, SEM is √[p(1−p)/(N−1)]. The corrected panel propagates the three-label count covariance through the correction. Neither bar is a fit error, a shot or round spread, or classifier training uncertainty. Missing, empty or single-shot points have no bar.

A large matrix condition number or small minimum singular value means the correction amplifies count noise. Preparation-label fractions are not transfer fidelity or proof of state purity.

## Next step

The dotted marker is the largest finite sampled corrected f, not a global optimum. Ties use the first acquired point; no finite point yields no peak. Choose a region of high corrected f and low e. If the marker is at the +0.10 µs endpoint, extend that endpoint while retaining the original overlap, then repeat. No writeback is proposed.

For `stirap_delay`, null clears generator channels, waveform or Readout selections, shots or relaxation wait. Unresolved required fields after overrides fail readiness before Run; missing saved readout calibration blocks acquisition. A completed Run can still save if Analyze fails.
