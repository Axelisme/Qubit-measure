**Readers:** operator, agent. Sweep flux reset gain at fixed hold duration and estimate excess residual excitation from matched IQ contrasts.

## Sequence

Optional **Reset** → **Pi pulse** on/off → **Tested reset** → the same **Pi pulse** on/off → **Readout**. The same program acquires all four combinations at each setting.

## Key settings

- Calibrate the π pulse and averaged-IQ readout at the operating point. The **Tested reset** template seeds the fixed hold duration from `$qubit.ext.flux_reset_length`; enter a calibrated value if absent.
- **Flux reset gain** starts at one point, zero gain. Set the range and count before Run. The sweep replaces only gain; gain is a normalized amplitude, not a flux axis. The tested reset's `post_delay` waits for the flux tail after the excursion; use a settling time measured at the operating point, for example with `fastflux/distortion`.
- **Relax delay** starts at 5 × T1 (150 µs if absent); enter a wait if T1 is invalid.

## Read the figures and choose a setting

Live shows the four before/after π branches versus gain (a.u.) on the principal IQ axis of the branch with no π pulses. This projection is not a population. Analyze shows I and Q separately to reveal branch motion a single projection can hide. Bars are shot SEM along the displayed axis: sample SD divided by √(retained shots), not fit error, round-to-round scatter, or uncertainty on the derived quantities. Mean-only files show no bars.

For complex means S[a,b] at before/after switch settings a,b, let d₀ = S[0,0]−S[0,1] and d₁ = S[1,0]−S[1,1]. The contrast panel shows r = Re(d₁·conj(d₀))/|d₀|²; the residual panel shows (1−r)/2. For ideal π pulses, stable readout and two states, p₁ = (1−r)/2 + r·p₀, where p₀ and p₁ are the post-reset excitation from g and e. The four means cannot separate p₀ and p₁. Treat the estimate as p₁ only if p₀ is small; a hot idle state can invalidate that assumption. This is neither fidelity nor a leakage measurement. Estimates outside [0,1] are not clipped.

Analyze selects the smallest finite estimate; ties use the first point in scan order. The vertical dashed line marks the chosen gain. Zero d₀ or nonfinite IQ, contrast or estimate excludes a point; if nothing qualifies, Analyze shows no candidate and proposes nothing.

The proposal updates only `$qubit.ext.flux_reset_gain`. Verify the accepted setting with the `flux_reset` template. Use `twotone/reset/check` for the averaged Rabi response or `singleshot/reset_check` to check the state distribution.

Shared GUI steps: [`zcu_lab/v2/operator-workflow.md`](../../../../operator-workflow.md).
