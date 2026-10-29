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

## Shared default and historical reference basis

The default oscillator cutoff is 80 with 20 retained DC eigenstates.
The Raman full-drive comparison required this increase from 15 levels.
The bounded reference and STIRAP regressions below passed before the change.
Existing working YAML files keep their own settings; updating this template does
not change a running environment. Review the basis in the working YAML and
restart after changing it.

The historical reference-only characterization used 80/15.
It compared cutoffs 60, 80 and 100 with 15, 20 and 25 retained
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

These reference results apply only to this circuit at zero flux, the specified
bath and short reference pulses. Cutoff 100 is a finite comparison, not an
infinite-basis oracle. The bounded STIRAP comparisons below provide separate
transfer, basis and full-drive evidence. They do not establish convergence for
stronger drives or sampled-envelope Raman. The later Raman checks below
support the 80/20 default only within their recorded controls.

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
independent, unbiased performance estimate.
The 80/20 default regression used independent seed 20261105 and the same
acquired reference controls. Each preparation again supplied 4096 shots,
with the first 2048 for training and the remaining 2048 for evaluation.
It did not select new controls or move shots between the halves.

| Observable | Approved threshold | 80/20 regression |
| --- | ---: | ---: |
| Minimum pooled Mahalanobis separation | ≥ 2 | 2.97982357 |
| Macro preparation-label error | ≤ 3% | 1.96940104% |
| Each preparation-label error, g/e/f | ≤ 6% | 0.390625% / 0.244140625% / 5.2734375% |
| Training confusion 2-norm condition | ≤ 1.5 | 1.05786181 |
| Training confusion minimum singular value | ≥ 0.8 | 0.951923881 |

The confusion matrix rows label requested ground, π01 and π03-then-π32
preparations. They include preparation contamination, readout jumps and
receiver noise. Correction returns preparation-label coordinates, not guaranteed
energy-state populations. Preserve corrected fractions below zero or above one;
clipping hides calibration uncertainty. Neither the error nor the separation
is pure readout fidelity or evidence of a hardware gate.

## STIRAP acquired-IQ characterization

The following acquired estimates use IQ and preparation labels only.
Operators, GUI, MCP and measurement Results must not read simulator truth.
The separate lab-author section below records environment-validation oracle
comparisons, not another source of measurement evidence.

The STIRAP controls came from acquired spectroscopy and Rabi calibration.
They are recorded observations, not Form defaults or reusable calibration for
another entry. The [STIRAP guide](../../v2/twotone/stirap/guide.md) explains
spectroscopy, reference preparation and saved-readout analysis.

| Control | Recorded value |
| --- | --- |
| Pump carrier, generator | 7241.848921606473 MHz, 1 |
| Stokes carrier, generator | 3111.1686681315896 MHz, 2 |
| Pump/Stokes base gains | 0.015 / 0.05 |
| Gaussian sigma, support per pulse | 0.2 us, six sigma = 1.2 us |
| Relax delay | 1 us |
| Acquired seed, shots per scan point | 20261106, 4096 |
| Reference seed | 20261105 |
| Reference training/evaluation shots | 2048 / 2048 per preparation label |
| Readout carrier, generator, ADC | 7460 MHz, 0, 0 |
| Readout gain, constant pulse, integration | 0.1, 0.4 us, 0.4 us |
| Readout phase, trigger offset, pre/post delay | 0 degrees, 0 us, 0 us |

The circuit, bath, loss, routing and receiver are those recorded above.
The receiver has zero ADC offset, no background response and no drift.
Numerical settings were cutoff 80, retained levels 15, rtol = 1e-7,
atol = 1e-9, max_step_ns = 0.05, bath_segment_ns = 10,
20 slow-envelope segments per period and projection_prune_floor = 1e-9.
The acquired basis comparisons changed only the stated basis sizes.
The explicit Engine budgets in the author-only comparisons below are separate
from these YAML tolerance settings.

Positive lead means Stokes starts first. The delay scan used nine requested
leads from -0.4 to +0.4 us in 0.1 us steps, with the base gains unchanged.
The amplitude scan fixed lead at +0.3 us and multiplied both gains by
0.5, 0.75, 1, 1.25 and 1.5. Sigma stayed at 0.2 us.
The detuning scan fixed lead at +0.3 us and used the Cartesian product of
Pump and Stokes offsets -0.5, 0 and +0.5 MHz, nine points in total.

These three acquired-IQ criteria are approved only for the recorded controls,
seeds and receiver. The f label denotes the pi03-then-pi32 reference preparation.

| Observable | Approved threshold | Observed result |
| --- | ---: | ---: |
| Corrected f at lead +0.2 / +0.3 / +0.4 us | Each ≥ 0.90 | 0.924389 / 0.969959 / 0.953925 |
| Corrected f at the five amplitude scales, lead +0.3 us | Each ≥ 0.94; range ≤ 0.04 | Minimum 0.955962, maximum 0.985752, range 0.029790 |
| Maximum absolute corrected-fraction difference, 80/15 versus 100/20 and 100/25, base gains and lead +0.3 us | ≤ 0.005 | 0.000254587 for each comparison |

The four negative-lead points gave corrected f from 0.174483 to 0.492034.
The detuning grid gave corrected f from 0.809268 to 0.998233.
These numbers describe the acquired two-dimensional response. There is no
approved minimum-transfer criterion for every detuning point.

Classification and confusion correction use the acquired three-reference
calibration. Analysis of a saved run reads that run's saved readout classifier
and training confusion, not the live library. The retained IQ was reanalyzed
with the public three-center analysis API, with raw and corrected fractions
unchanged from the original characterization.
Corrected fractions are preparation-label coordinates, not certified
energy-state populations. Keep negative and above-one estimates.
Neither a transfer plateau nor its confusion correction establishes hardware
gate fidelity.

## Bounded 80/20 STIRAP acquisition regression

At the recorded lead +0.3 us working point, 80/20 gave raw g/e/f fractions
0.036376953125 / 0.03515625 / 0.928466796875.
The corrected fractions were -0.0000180691 / 0.0328592004 / 0.9671588687.
The maximum absolute corrected-fraction difference from the saved 80/15
single-point run was 0.0002545872434, below the approved 0.005 limit.
The controls, seed 20261106, 4096 shots and original reference calibration
stayed fixed. This regression did not repeat or revalidate any scan grid.
The historical STIRAP numbers above retain their original 80/15 scope.

## Acquisition wall time

One public legacy `ModularProgramV2.acquire` call at the same STIRAP lead
+0.3 us point took 1.290324 s at 80/15 and 20.510185 s at 80/20.
These are single samples, not a performance threshold or a stable speed ratio.
The timer starts immediately before acquire and stops when it returns.
It includes simulator admission, numerical evolution, ADC sampling and buffer
collection. It excludes imports, environment and program construction, public
template capture, IQ conversion, analysis, plotting and file I/O.
Both calls used the carriers, Gaussian sigma 0.2 us, gains 0.015/0.05,
seed 20261106, 4096 shots and readout settings in the STIRAP table above.

Each basis ran once in a fresh process and environment, with no acquire warmup.
The existing source-namespaced Numba disk cache was retained.
The 80/20 call ran first with 178 cache files present; the later 80/15 call
started with 214. Compilation and disk-cache reuse can therefore contribute
to the difference. Do not treat these samples as isolated basis-compute costs
or predict a whole-grid duration by multiplying them.
The host was an Intel Core Ultra 7 258V, 8 logical CPUs, Linux x86_64
7.2.11-arch1-1, Python 3.13.11 and NumPy 2.4.6.
OMP, OpenBLAS, MKL and Numba thread limits were all 1.
The headless run used Qt offscreen and Matplotlib Agg, with displays unset.
No repeated benchmark or performance scan was run.

## Raman acquired-IQ characterization

The recorded 12 points used two-photon detunings -50, 0 and +50 MHz with
requested gains 0, 0.3, 0.6 and 0.9 at seed 20261106 and 4096 shots per point.
The carrier center came from acquired spectroscopy as
(f03 - f32) / 2 = 2065.3401267374416 MHz, not a Form default.
The source waveform was sqrt(sin(pi*t/T)), T = 0.1 us, with 101 samples.
Compilation produced a 0.09895833333333333 us drive, actual nonzero gains
0.29997558444729294, 0.5999816883354697 and 0.8999877922236464,
and carriers 2040.340126991272, 2065.3401260375977 and
2090.3401279449463 MHz. Readout began at 0.10230654761904762 us
in a 0.5045572916666667 us compiled timeline.
The circuit, bath, receiver and reference calibration stayed fixed.

| Acquired preparation-label observable | Approved threshold | Recorded result |
| --- | ---: | ---: |
| Resonant gain 0.9 raw / corrected f | Each ≥ 0.45 | 0.514892578125 / 0.535659660097 |
| Increase over resonant zero gain, raw / corrected f | Each ≥ 0.45 | 0.510986328125 / 0.533112541518 |
| All zero-drive and ±50 MHz points, maximum raw f / abs corrected f | Each ≤ 0.01 | 0.00390625 / 0.00254711857834 |
| All 12 points, maximum abs corrected e-label | ≤ 0.002 | 0.000387550820873 |

The original 4096-shot runs did not save a classifier. These estimates apply
the same receiver's acquired and Apply-validated calibration offline.
They characterize those IQ records, not the saved-run contract.
Normal GUI Run, Save and native Load separately validate analysis from saved cfg.
Correction retains negative values and returns preparation-label coordinates.
The e label is not all higher-state leakage. Transfer is not gate fidelity.
The [Raman guide](../../v2/twotone/raman_gf/guide.md) describes that workflow.

## STIRAP environment oracle, lab-author only

Only the simulator/environment-validation owner uses the oracle in this section.
The owner reconstructed the recorded Gaussian programs through the public
program-template and simulation interfaces. Internal density matrices and
populations stayed outside the experiment, GUI, MCP and measurement Results.
Operators must use the acquired-IQ workflow above instead.

Production mixed-route diagnostics reported multitone_rwa, shaped_sambe and t4.
Do not describe the acquire path as full DOP853.
The full-drive comparison used public Engine kernel injection to decline
approximate routes while retaining production DOP853 and idle T4.
Diagnostics reported three dop853 segments and one t4 segment at the endpoints.
This validation did not change solver policy, Engine defaults or physical YAML.

Basis and solver differences below are maximum absolute differences over the
complete retained-state population vector. The comparison zero-pads the shorter
vector.
All full-drive endpoint and sampled-peak basis comparisons used Engine budget
1e-5. At lead +0.2 us, the endpoint was 1404.389880952381 ns.
At lead +0.3 us, it was 1504.3712797619048 ns.
These are absolute compiled timeline times, not the requested pulse lengths.

| Observable | Approved threshold | Observed result |
| --- | ---: | ---: |
| Full-drive basis difference at tested endpoints and sampled peak | ≤ 1e-4 | Endpoint maximum 9.66193e-6; sampled peak 3.81350e-5 |
| Lead +0.2 us, full DOP853 budget 1e-5 versus 1e-6 | ≤ 1e-6 | 2.08838e-9 |
| Lead +0.2 us endpoint, mixed budget 1e-3 versus full budget 1e-5; lead +0.3 us sampled peak, both budgets 1e-5 | Each ≤ 5e-4 | Endpoint 2.06905e-4; sampled peak 3.48006e-7 |
| Lead +0.3 us sampled maximum level-3 population and its difference from lead -0.3 us | Maximum ≤ 0.10; difference ≥ 0.35 | 0.0831593 versus 0.5170004; difference 0.4338411 |

The endpoint basis difference at lead +0.2 us compared 80/15 with 100/20
and was 9.661931191305706e-6.
At lead +0.3 us, the 80/15 versus 100/25 endpoint difference was
3.89238629409927e-6.
At 752.1856398809524 ns, the positive-lead sampled peak, the same full-drive
80/15 versus 100/25 comparison gave 3.813497585855585e-5.
Full-drive level-3 populations there were 0.0831593135 and 0.0831423936.

The lead +0.2 us endpoint comparison at 80/15 used production mixed budget
1e-3 and high-precision full-drive budget 1e-5.
Its difference, 2.069049176611007e-4, combines route and budget effects.
It is not a same-budget solver-route error estimate.
The lead +0.3 us sampled-peak comparison at 80/15 used budget 1e-5 on both
sides and gave 3.480064294514307e-7.
Do not extend that sampled-peak result to the endpoint.
Each approved limit applies only to its recorded controls, time and budgets,
not to arbitrary production settings.

The finite-time level-3 comparison used the mixed route at budget 1e-5,
80/15, base gains, sigma 0.2 us and leads +0.3 and -0.3 us.
Both sequences used these seven absolute timeline times in ns:
451.31138392857144, 601.7485119047619, 752.1856398809524,
902.6227678571429, 1053.0598958333333, 1203.4970238095239
and 1504.3712797619048.
The sampled maximum is not a bound on the continuous-time maximum.
The full-drive check at the positive-lead sampled peak does not turn the
other sampled times into full-drive comparisons.

These historical STIRAP checks supported 80/15 at the tested working points.
The later 80/20 regression did not repeat the historical seven-item acceptance.
At lead +0.3 us, full-drive 80/20 versus the retained 100/25 results gave
maximum zero-padded population differences 7.54545280e-6 at the
1504.3712797619048 ns endpoint and 1.64302745e-5 at the already sampled
752.1856398809524 ns peak. Both sides used effective Engine budget 1e-5.
Both differences passed the approved 1e-4 limit. Only the missing 80/20 side
was evolved; the peak time and 100/25 records stayed fixed.
This does not validate the entire STIRAP grid at 80/20.
The short-reference checks above do not replace these driven comparisons.
100/25 remains a finite reference. The basis and solver evidence does not
establish convergence at every point of any scan grid.
Do not extend these thresholds to other controls, seeds, times, budgets,
stronger drives, longer sequences, Raman or another circuit, loss model or
receiver. An early pilot with sigma 0.05 us and gains 0.06/0.20 is outside
this plateau acceptance scope.

## Raman environment oracle, lab-author only

The same 12 compiled Raman controls were evolved through public Engine.evolve.
Driven segments selected full-drive non-RWA dop853; idle segments used t4.
The shaped_sambe candidate was rejected for the nine driven points.
This is not acceptance of an approximate driven solver.
YAML rtol/atol changes that gave zero difference are not DOP853 convergence
proof. The comparisons below changed the effective Engine budget instead.

| Complete population comparison, shorter vector zero-padded | Approved threshold | Maximum absolute difference |
| --- | ---: | ---: |
| 80/20 budget 1e-3 versus 1e-7 | ≤ 1e-4 | 6.15992704119e-5 |
| 80/20 budget 1e-5 versus 1e-7 | ≤ 1e-6 | 6.10800183876e-7 |
| 80/20 versus 100/25, both budgets 1e-5 | ≤ 1e-4 | 2.33183304376e-5 |

80/15 versus 100/25 gave 2.27051325114e-3 at budget 1e-5 on both sides.
It failed the unchanged 1e-4 basis limit. This is why the shared default now
retains 20 levels. All three bounded reference/STIRAP regressions passed
before the default changed. The 100/25 comparison remains a finite reference.
These results cover only the recorded 12 points, waveform, compiled timing,
receiver, seed and budgets. They do not establish convergence at longer pulses,
higher gains, continuous detuning/area ranges or other circuit parameters.
The oracle is author-only evidence, not a measurement Result or calibration.

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
