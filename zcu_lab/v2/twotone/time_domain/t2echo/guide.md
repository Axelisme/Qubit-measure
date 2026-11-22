**Readers:** operator, agent. Measure spin-echo coherence time T₂E after refocusing slow dephasing.

## Sequence and key settings

Optional **Reset** → calibrated **Pi/2 Pulse** → half delay → calibrated **Pi Pulse** → half delay → final π/2 → calibrated **Readout**. Sweep the total free-evolution delay. T₂E is the decay time of the echo envelope and is often longer than Ramsey T₂*.

- **Delay** starts at `0…1.5 × qubit.t2e` (µs). Span about **3 × T₂E** for a well-resolved decay. Too short a scan leaves the tail unseen and makes the fit depend on extrapolation.
- **Detune ratio (fringes/step)** sets the final π/2 phase ramp; requested MHz is ratio ÷ Step (µs). For a fringe fit, use at least **10 points per period**, so Step ≤ `1/(10f)` µs for fringe frequency f in MHz. Increase **Expts** with the window; sparse points can hide cycles or show an aliased fringe.
- **Fit method** defaults to `decay`. For a standard echo use ratio 0. Choose `fringe` for a nonzero source detune; **Fit phase offset (fringe only)** frees the phase. Zero source detune forces decay. A fringe request without captured cfg detune or a finite saved actual detune fails Analyze; decay remains available.
- **Relax delay** starts at `5 × qubit.t1` (150 µs when absent). Check both pulses and readout timing before Run.

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the plots

Live shows rotated mean IQ amplitude (a.u.), not population, against total delay in µs. Its detune title reports the applied ramp. Analyze plots that IQ projection with the decay or fringe fit and omits the first measured point from the fit. The lifetime uncertainty is a fit error; neither plot has pointwise shot SEM.

R² compares residuals with data variation. Normalized residual RMS gives a residual scale relative to the measured signal span. Review the decay envelope and fit uncertainty, especially when it barely falls in the measured window.

## Next step

Writeback proposes `qubit.t2e` and an independent bias sample when available; it does not change RF frequency or module slots. T₂/sample proposals are preselected only with positive finite T₂, finite nonnegative relative error ≤ 0.5, R² ≥ 0.9 and normalized residual RMS ≤ 0.1. Failed, invalid or unavailable checks leave candidates unselected for manual review.

For `t2echo`, null disables **Reset** or clears the named Readout, π or π/2 pulse, delay Stop/Expts, detune ratio, Reps or Rounds. Required blanks left after overrides fail readiness before Run. Null for **Fit method** or **Fit phase offset (fringe only)** fails at Analyze.
