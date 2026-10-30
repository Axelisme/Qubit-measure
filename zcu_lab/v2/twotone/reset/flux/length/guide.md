**Readers:** operator, agent. Scan flux reset hold length at fixed gain and review a four-branch residual excitation estimate.

## Preparation and sequence

Calibrate the qubit pi pulse and averaged-IQ readout at the operating point before this scan.
Each length point measures four branches in one program with the same readout.
Each repetition applies optional reset, the before-reset pi switch, the tested flux reset, the after-reset pi switch, then readout.
Both switches use the same calibrated pi pulse. Each switch has off=0 and on=1.
Saved complex IQ has axes before pi, after pi, length, in that order.
The saved length axis contains the hardware-quantized hold durations in us.

The tested reset uses `reset/pulse` on `$qubit.wiring.lo_flux_ch`, with frequency 0 MHz and a const waveform.
Its fixed gain starts from `$qubit.ext.flux_reset_gain` through the `flux_reset` template.
Review that gain from the gain scan before calibrating duration. A gain proposal alone does not prove effective reset.
Missing required sources remain Invalid. Enter a fixed gain if it has not been saved.
Open and Reset capture the module sources and fill editable starting values.
Pi pulse uses `$qubit.module.x180` when present, otherwise the pi-pulse template. Verify its calibration before Run.
Optional reset uses `$qubit.module.reset`; a missing slot leaves it disabled.
Readout uses `$resonator.module.readout`, or the readout template if that slot is absent.

## Scan settings

The length range starts at a single 0.1 us point. Set start, stop and point count for the permitted hold-length range before Run.
Run replaces only the tested pulse's hold length. It keeps the gain and pre/post delays unchanged.
The tested pulse must retain its const waveform and zero frequency.
Its `pulse_cfg.post_delay` is the time in us to wait after returning from the flux excursion for the flux tail to settle.
0 means no wait. On hardware, measure the flux tail, for example with `fastflux/distortion`, and set this delay long enough for settling.

Reps and rounds start at 1000 and 10. Each branch/length point receives that many repetitions per round.
Relax delay starts at five times the captured T1. Missing T1 uses 30 us as a starting time scale, giving 150 us.
Invalid T1 leaves the derived field empty. Review the wait between repetitions rather than treating it as a calibrated reset time.

## Reading the contrasts

The live plot shows I for the four branches, not a population projection.
Analysis plots raw I and Q, the contrast ratio r and the residual estimate against quantized length.
Let S[a,b] be mean complex IQ with before-pi switch a and after-pi switch b.
The after-pi differences are `d0=S[0,0]-S[0,1]` and `d1=S[1,0]-S[1,1]`.
Analysis calculates `r=Re(d1*conj(d0))/|d0|²` without PCA or external IQg/IQe inputs.

Under ideal calibrated pi pulses, the same readout and a two-state approximation, `d0=(1-2*p0)*(IQg-IQe)` and `d1=(1-2*p1)*(IQg-IQe)`.
Here p0 is excitation after reset starting from g, and p1 is excitation after reset starting from e.
The exact relation is `p1=(1-r)/2+r*p0`.
Four mean IQs alone cannot identify p0 and p1 separately.
This experiment assumes the reset hardly heats g, so p0 is small, and reports `residual_estimate=(1-r)/2`.
In general this estimate is `(p1-p0)/(1-2*p0)`, not absolute p1.
A thermally excited idle point, such as fluxonium near half flux, invalidates the p0≈0 approximation.
The fixed offline model used for validation has idle first-excited-state population about 0.358.
Only after an initial reset makes p0 small can the estimate represent absolute residual excitation.
The pi, readout and two-state assumptions must also hold.
Offline validation checks run, save, restore, analyze, PNG output and selection or no-solution handling.
It does not establish physical reset effectiveness.
Do not use model populations to calibrate the estimate or change the four-branch formula, epsilon or selection.
Imperfect pi pulses, readout changes and failure of the two-state approximation can bias it.
It cannot exclude leakage or establish reset fidelity.
Estimates outside [0,1] stay visible. Analysis does not clip noise or assumption failures into a probability.

## Reviewing the proposal

Epsilon is an editable analysis option, default 0.05. Change it in the analysis pane and run Analyze again to review another threshold without acquiring new IQ.
Analysis selects the shortest quantized length with residual estimate ≤ epsilon, provided every longer measured finite point also passes.
It compares lengths, not scan indices, so reversed or non-sorted scans use the same rule.
An isolated early low estimate does not qualify when a longer finite point fails.
The horizontal dotted line marks epsilon. The vertical dashed line marks the selected length, when one exists.
Zero d0 makes the ratio undefined. Nonfinite IQ, contrasts or estimates also exclude a point from selection and the longer-point check.
No extra near-zero contrast threshold is applied.
Excluded or unmeasured points do not certify behavior between the measured lengths or beyond the scan endpoint.
If no finite or qualifying candidate remains, the plot says so and analysis produces no writeback proposal.

Review the contrasts and assumptions before accepting the `$qubit.ext.flux_reset_length` proposal in us.
Analysis only proposes this ext value. It does not Apply it or change the current point automatically.
A low estimate alone does not prove effective reset. Verify an accepted setting with `twotone/reset/check` or `singleshot/reset_check` using the `flux_reset` template.
