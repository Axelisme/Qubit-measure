**Readers:** operator, agent. Scan a fixed-length flux reset's gain and review a four-branch residual excitation estimate.

## Preparation and sequence

Calibrate the qubit pi pulse and averaged-IQ readout at the operating point before this scan.
Each gain point measures four branches in one program with the same readout.
Each repetition applies optional reset, the before-reset pi switch, the tested flux reset, the after-reset pi switch, then readout.
Both switches use the same calibrated pi pulse. Each switch has off=0 and on=1.
The saved complex IQ has axes before pi, after pi, gain, in that order.

The tested reset uses `reset/pulse` on `$qubit.wiring.lo_flux_ch`, with frequency 0 MHz and a const waveform.
The gain scan uses a.u.; it does not convert gain to flux.
Open and Reset capture the module sources and fill editable starting values.
Pi pulse uses `$qubit.module.x180` when present, otherwise the pi-pulse template. Verify its calibration before Run.
Optional reset uses `$qubit.module.reset`; a missing slot leaves it disabled.
Readout uses `$resonator.module.readout`, or the readout template if that slot is absent.
The tested reset starts from the `flux_reset` template with fixed length from `$qubit.ext.flux_reset_length` in us.
Missing required sources remain Invalid. Enter a duration to calibrate gain before a reset length has been saved.

## Scan settings

The gain range starts at a single zero-gain point. Set start, stop and point count for the permitted local-flux range before Run.
Run replaces only the tested pulse's gain. It keeps the fixed length and timing unchanged.
The tested pulse must retain its const waveform and zero frequency.
Its `pulse_cfg.post_delay` is the time in us to wait after returning from the flux excursion for the flux tail to settle.
0 means no wait. On hardware, measure the flux tail, for example with `fastflux/distortion`, and set this delay long enough for settling.

Reps and rounds start at 1000 and 10. Each branch/gain point receives that many repetitions per round.
Relax delay starts at five times the captured T1. Missing T1 uses 30 us as a starting time scale, giving 150 us.
Invalid T1 leaves the derived field empty. Review the wait between repetitions rather than treating it as a calibrated reset time.

## Reading the contrasts

The live plot shows I for the four branches, not a population projection.
Analysis plots the raw I and Q, the contrast ratio r and the residual estimate against gain.
Let S[a,b] be mean complex IQ with before-pi switch a and after-pi switch b.
The after-pi differences are `d0=S[0,0]-S[0,1]` and `d1=S[1,0]-S[1,1]`.
Analysis calculates `r=Re(d1*conj(d0))/|d0|²` without PCA or external IQg/IQe inputs.

Under ideal calibrated pi pulses, the same readout and a two-state approximation, `d0=(1-2*p0)*(IQg-IQe)` and `d1=(1-2*p1)*(IQg-IQe)`.
Here p0 is excitation after reset starting from g, and p1 is excitation after reset starting from e.
The exact relation is `p1=(1-r)/2+r*p0`.
Four mean IQs alone cannot identify p0 and p1 separately.
This experiment assumes the reset hardly heats g, so p0 is small, and reports `residual_estimate=(1-r)/2`.
In general this estimate is `(p1-p0)/(1-2*p0)`, not absolute p1.
Imperfect pi pulses, readout changes and failure of the two-state approximation can bias it.
It cannot exclude leakage or establish reset fidelity.
Estimates outside [0,1] stay visible. Analysis does not clip noise or assumption failures into a probability.

## Reviewing the proposal

Analysis selects the smallest finite residual estimate. Ties use the first point in scan order.
The dashed line marks the selected gain. Review the contrasts and assumptions before accepting the `$qubit.ext.flux_reset_gain` proposal.
Zero d0 makes the ratio undefined. Nonfinite IQ, contrasts or estimates also exclude a point from selection.
If no finite candidate remains, the plot says so and analysis produces no writeback proposal.
A low estimate alone does not prove effective reset. Verify the accepted setting with `twotone/reset/check` or `singleshot/reset_check` using the `flux_reset` template.
