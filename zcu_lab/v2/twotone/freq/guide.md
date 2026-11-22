**Readers:** operator, agent. Locate a qubit transition and fit its RF center and linewidth from resonator IQ contrast.

## Sequence and key settings

Optional **Reset** → optional **Init Pulse** → frequency-swept **Probe Pulse** → resonator **Readout**. Leave Init Pulse off for a ground-state search. To probe after preparation, choose a calibrated pulse; its carrier stays fixed while the probe sweeps.

- Bind `resonator: $qubit.resonator` and select calibrated readout from `$resonator.module.readout`. The old qubit readout slot is not used. Inspect readout timing and probe gain and duration even if a module is bound. The probe starts from a spectroscopy template, not X180.
- **Probe RF frequency** starts at `$qubit.freq ± 20 × $qubit.kappa` in MHz. For a first search enter the approved RF window directly, then narrow around an observed peak. Choose a step much smaller than the linewidth before adopting a fitted FWHM; a line in one or two samples only locates a candidate.
- **Model type** `lor` uses the Run observable; `sinc` uses a signed IQ projection for visible sidelobes. **Min SNR** gates both prominence and height above the median in noise σ (default 5). **Expected center** selects the highest local maximum within the larger of two median steps or half its half-prominence width; blank selects the most prominent candidate. An unmatched window or fewer than six samples fails Analyze.

See [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md) for the shared GUI steps.

## Read the spectrum

Run shows the magnitude after subtracting the median complex IQ, in a.u., versus probe RF frequency in MHz. A peak indicates a change in readout response, not population. There are no pointwise shot SEM bars; center and FWHM uncertainties are fit errors. Candidate SNR measures height above the median in noise units, not fit quality.

Analyze shades the fitted interval, with a zoom if it spans at most half the scan. The window starts at ± six half-widths about the selected peak, clips at neighboring midpoints and expands if needed to include six samples. Gray triangles mark other candidates; the dotted line is the fitted center. Check the curve against both shoulders. A candidate flagged `truncated` has a half-prominence crossing outside the scan. Widen or recenter before using its width.

## Next step

Only adopt writeback if the feature is the intended qubit line. It proposes fitted `qubit.freq` and `qubit.kappa` (FWHM), plus an independent frequency sample when Run bias evidence is available. Module slots stay unchanged. For a prepared 3–2 or a ground-state 0–3 scan, record the measured carrier separately rather than writing it to a 0–1 calibration.

For `twotone_spectrum`, null disables **Reset** or clears the named Readout, Probe Pulse, Gain, Length, Reps or Rounds. Unresolved required blanks after overrides fail readiness before Run. Null **Model type** fails at Analyze; omit it to keep the existing choice.
