# Integer-fluxonium field guide

**Readers:** operator, lab-author, agent. Interpret the synthetic basis and readout checks without treating model inputs as measured calibration.

## Circuit, bath and routing

The circuit uses EJ/EC/EL = 6.78/1.47/0.22 GHz, Φ/Φ0 = 0 and a 20 mK qubit bath.
These are Mencia 2024 Device D inputs to a simplified fluxonium model.
The fixed circuit produces synthetic f03 ≈ 7.24184 GHz, while the literature
reports 6.86 GHz. Keep the circuit fixed rather than tuning it to match the
reported line. That synthetic frequency is not calibration and is not a
Form or entry default. Measure 0–1 and 0–3 from ground, calibrate π03, then
measure 3–2 after π03 with the optional Init Pulse in `twotone/freq`.

The explicit loss model has q_cap = 100000, q_ind = null, xqp = 0 and aphi = 0.
There is no drift. The DC source starts at zero with offset_phi0 = 0.
Its synthetic scale is 1 Φ0/A. Each acquisition point starts from the retained
DC Gibbs state; it does not inherit the previous shot's final state.

Pump and Stokes bind generators 1 and 2 to the full charge operator.
Both use charge-drive coefficient 1 GHz/gain.
Readout binds generator 0 and ADC 0. No fast-flux route is enabled.
The schema can represent a flux route, but generator 2 already belongs to
Stokes in this template. Unknown or colliding routes fail before evolution.

## Reference basis comparison

The default oscillator cutoff is 80 and the retained DC space has 15 eigenstates.
The characterization compared cutoffs 60, 80 and 100 with 15, 20 and 25 retained
levels. Its finite reference was 100/25. It held the controls, receiver and
seed 20261029 fixed across comparisons. It used short 100 ns constant reference
pulses for ground, π01 and π03 followed by π32. It did not recalibrate the pulse
area at each basis.

The following seven absolute-difference thresholds were approved for this
reference-only comparison. Common states exist in both retained spaces.
The population comparison zero-pads the shorter vector to include discarded
high-state weight. IQ differences use the Euclidean norm of the two-dimensional
conditional mean per sample, in ADC units.

| Observable | Threshold | 80/15 maximum difference from 100/25 |
| --- | ---: | ---: |
| f01, f12, f03 | ≤ 1 kHz | 0.156853 kHz |
| abs(n01), abs(n02), abs(n03), abs(n23), abs(n12) | ≤ 1e-8 | 6.20743e-9 |
| g/e/f dressed cavity frequency | ≤ 0.1 kHz | 0.0109631 kHz |
| All common-state cavity frequencies | ≤ 1 kHz | 0.109731 kHz |
| g/e/f conditional mean IQ vector | ≤ 0.001 ADC | 4.57034e-5 ADC |
| All common-state conditional mean IQ vectors | ≤ 0.03 ADC | 0.0170237 ADC |
| All acquire populations, environment-validation oracle only | ≤ 1e-5 | 2.06786e-6 |

The environment-validation owner alone uses the population oracle for this
comparison. Measurement analysis uses acquired IQ and preparation labels,
not internal populations.

80/15 passed all seven checks and was the smallest tested passing combination.
Cutoff 60 failed with frequency difference 11.7996 kHz, g/e/f cavity difference
0.942017 kHz and acquire-population difference about 1.93e-4.
At cutoff 80, increasing levels 15 to 20 changed the reference acquire populations
by at most 3.93361e-8. Near-zero parity matrix elements use absolute tolerances;
relative error would misrepresent n02. High-state IQ uses its own tolerance,
not the tighter g/e/f limit.

These results apply only to this circuit at zero flux, the specified bath and
short reference pulses. Cutoff 100 is a finite comparison, not an infinite-basis
oracle. Strong dual-line STIRAP and sampled-envelope Raman still need their own
basis, transfer, leakage and full-drive checks. If 15 levels are insufficient,
raise the environment default only after reviewing new numerical evidence and
approving the replacement thresholds. Update this guide with that scope.

## Readout settings and five checks

| Setting | Value |
| --- | --- |
| Bare cavity and reference readout carrier | 7.46 GHz |
| Cavity charge coupling | 0.10 GHz |
| External κ/(2π) | 15 MHz |
| Internal κ/(2π) | 0 MHz |
| Residual cavity thermal occupation | 0 |
| Constant pulse and acquisition duration | 400 ns |
| Incident field | 0.1 sqrt(photons/ns) |
| Readout input_per_gain | 1 sqrt(photons/ns)/gain |
| ADC gain | 1000 |
| Diagonal noise PSD | 400 ADC² ns per quadrature |
| Off-diagonal noise PSD | 0 |
| Propagation delay and receiver phase | 0 ns, 0 degrees |

The 0.10 GHz cavity coupling is not the 1 GHz/gain microwave calibration.
The linewidth split is an explicit synthetic choice, not a measured
external/internal decomposition.
Static cavity response includes the full native circuit space and returns
frequencies in retained DC energy order. It is not a three-state charge
approximation. Rates and dynamics still use the retained space.

The selection compared 36 combinations of coupling, pulse and receiver.
Each preparation had 4096 synthetic shots. The first 2048 trained a nearest
centroid IQ classifier. The remaining 2048 evaluated it.
The minimum separation uses pooled within-preparation covariance.
The selected 80/15 settings gave these results with seed 20261029:

| Observable | Approved threshold | Selection-dataset result |
| --- | ---: | ---: |
| Minimum pooled Mahalanobis separation | ≥ 2 | 2.53274 |
| Macro preparation-label error | ≤ 3% | 95/6144 = 1.5462% |
| Each preparation-label error | ≤ 6% | Maximum 83/2048 = 4.0527% |
| Training confusion 2-norm condition number | ≤ 1.5 | 1.06804 |
| Training confusion minimum singular value | ≥ 0.8 | 0.944051 |

The evaluation half participated in selecting the settings. It is not an
independent, unbiased performance estimate. Three-state readout acceptance
must repeat these five checks with a fixed seed other than 20261029.

The confusion matrix rows label requested ground, π01 and π03-then-π32
preparations. They include preparation contamination, readout jumps and
receiver noise. Correction returns preparation-label coordinates, not guaranteed
energy-state populations. Preserve corrected fractions below zero or above one;
clipping hides calibration uncertainty. Neither the error nor the separation
is pure readout fidelity or evidence of a hardware gate.

## Solver and unsupported scope

Reference runs used rtol = 1e-7, atol = 1e-9, max_step_ns = 0.05,
bath_segment_ns = 10 and 20 slow-envelope segments per period.
Comparisons at rtol/atol of 1e-6/1e-8, 1e-8/1e-10 and 1e-10/1e-12
gave zero change in the recorded reference observables.
Those runs selected single_tone_rwa and t4, not DOP853.
The zero difference shows tolerance insensitivity for those routes only.
It does not bound RWA error or establish full-drive integration convergence.

Only terminal cavity readout is supported. An earlier readout-generator pulse
fails rather than modeling cavity memory, Stark shifts or backaction.
The characterization did not scan resonator photon cutoff, nonlinear response,
MIST, flux away from zero, fast flux, long sequences, receiver background or
other circuit parameters. Cavity thermal occupation zero does not mean the
qubit bath is zero-temperature.

Keep simulator truth and replay archives on the environment-validation side.
GUI, MCP and operator analysis consume IQ, preparation labels and public
diagnostics. They must not consume density matrices or model-internal populations.
All results here are synthetic and are not hardware acceptance criteria.
