**Readers:** operator, agent. Sweep the gain of a fixed-frequency, fixed-length drive and fit the oscillation to calibrate π and π/2 gains.

## Sequence and key settings

Optional **Reset** → optional **Init Pulse** → gain-swept **Rabi Pulse** → calibrated **Readout**. Leave Init Pulse off for ground-state preparation; otherwise choose a calibrated fixed preparation. Use a measured drive carrier and check readout timing. A prior π calibration is not needed if the pulse and sweep are set directly.

- **Pulse gain › Stop** starts at `min(1.0, 1.2 × stored X180 gain)`; enter it when the stored gain is unavailable. Span enough gain to see at least one period, roughly two π gains.
- The editable X180 copy starts at `1.5 × X180 length × X180 gain` in µs. This needs finite positive length and finite gain in (0, 1]; invalid input leaves length blank. Without X180, the spectroscopy template starts at gain 1 and length 1 µs. A saved arbitrary waveform keeps its asset and duration; select an editable shape to change the duration.
- **Relax delay** starts at `5 × qubit.t1` (150 µs if T₁ is absent). **Skip points** drops leading gain points before fitting; use it only if those points are distorted.

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the plots

Both plots show mean IQ projected along the trace's principal direction (a.u.; I if undefined), not population, versus gain. The live title counts retained shots per point. One bar is one shot SEM: retained-shot sample SD along that axis divided by √(retained shot count). It is not shot spread, round-to-round scatter or fit error, and does not weight the fit. Mean-only files still load and fit without bars.

Analyze overlays a cosine. π and π/2 are the gains for half and quarter of a Rabi cycle; their ± values are fit standard errors. R² compares residuals with the variation of included points. It is neither a selection threshold nor gate fidelity. Check the curve across a full period before using the markers.

## Next step

Writeback proposes `pi_amp`/`pi2_amp` pulse copies with fitted gains, separate `qubit.module.x180/x90` assignments, and `qubit.ext.pi_product = fitted π gain × source pulse length`; it needs acquired Run settings.

### Prepared-state amplitude Rabi

For a prepared 3–2 scan, record the gain with its carrier, channel and duration instead of replacing your 0–1 pulses.

For `amplitude_rabi`, null disables **Reset**. Null **Readout** or **Rabi Pulse** fails during selection, before overrides. Null frequency, length, `gain_range` (both endpoints), Expts, Reps or Rounds leaves a blank; unresolved required blanks fail readiness before Run. Null **Skip points** fails at Analyze.
