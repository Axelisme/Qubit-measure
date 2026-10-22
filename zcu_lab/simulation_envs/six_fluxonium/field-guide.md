# Six-fluxonium settings guide

**Readers:** operator, lab-author, framework-dev. Read physical units, per-qubit parameters and approximation limits before changing this environment.

## Files and components

`default.yaml` uses `zcu.six-fluxonium` version 1.
`entry-template.yaml` uses `zcu.entry-template` version 1.0.
The template creates Q1 through Q6, R1 through R6, one shared `flux_bias`
current source and group `MG123456`. Each qubit references `flux_bias`.
Alias `flux_dev` binds directly to `flux_bias`. Tone i follows the group's
ordered resonator list R1 through R6 and its corresponding qubit list Q1 through Q6.
Group ADC wiring follows the same order. Resonator single-tone wiring is a
separate field, not a source from which the mux form infers ADCs.

| Route | Generator | ADC |
| --- | --- | --- |
| Mux readout | 0 | 0 through 5, in R1 through R6 order |
| Charge drive Q1 through Q6 | 1 through 6 | None |
| Shared pulse readout | 13 | 6 |

The top-level settings declare `mux_gen_ch: 0`, `pulse_gen_ch: 13` and
`pulse_ro_ch: 6`. Top-level `dc_source` holds the shared device's name, unit,
limits, initial current and ramp step. Each qubit's `dc_bias` contains only
`offset_phi0` and `flux_per_unit`, not a per-qubit source. Each nested Q1 through
Q6 value keeps `format`, `version`, `chip`, `drive_line`, `readout` and `numerics`.
Its readout generator is 0 and its ADC is its mux tone index.
The schema rejects per-qubit sources and `flux_line` sections. There is no
legacy schema fallback. Generators 7 through 12 are not configured.
Each resonator component instead declares
pulse wiring `ch: 13`, `ro_ch: 6`. `onetone/range_find` uses that pulse route.

## Circuit parameters

EJ, EC and EL are H/h in GHz. Cavity frequency also uses GHz in physical YAML;
entry resonator frequency uses MHz. Qubit `freq` remains null until measured.

| Qubit | EJ | EC | EL | Resonator | Bare frequency, GHz |
| --- | ---: | ---: | ---: | --- | ---: |
| Q1 | 4.0 | 1.0 | 1.0 | R1 | 7.00 |
| Q2 | 4.5 | 0.9 | 0.8 | R2 | 7.10 |
| Q3 | 5.0 | 1.1 | 0.6 | R3 | 7.20 |
| Q4 | 3.5 | 1.0 | 1.2 | R4 | 7.30 |
| Q5 | 6.0 | 0.8 | 0.5 | R5 | 7.40 |
| Q6 | 4.2 | 1.2 | 0.9 | R6 | 7.50 |

Each qubit uses q_cap 1e6, q_ind 1e8, xqp 2e-7 and aphi 2e-6 at 0.05 K.
Each resonator uses external linewidth 1.0 MHz and internal linewidth 0.1 MHz,
both kappa divided by 2*pi. The full-charge coupling coefficient is 0.03 GHz.
There is no qubit-qubit coupling or crosstalk matrix. All six circuits respond
to the same live DC current.

## DC and control units

The single global DC FakeDevice `flux_bias` has unit A, limits -0.02 to 0.02 A,
initial current 0 A and ramp step 1e-6 A. Each qubit's reduced DC flux equals
`offset_phi0 + flux_per_unit * current_A`. A current change therefore moves all
six operating points, with different offsets and periods. Startup applies the
initial current once. Acquisition reads the live device value.

| Qubit | offset_phi0 | flux_per_unit, Phi0/A | Current per period, mA |
| --- | ---: | ---: | ---: |
| Q1 | 0.50 | 100 | 10 |
| Q2 | 0.45 | 80 | 12.5 |
| Q3 | 0.55 | 120 | 8.333333 |
| Q4 | 0.40 | 90 | 11.111111 |
| Q5 | 0.60 | 110 | 9.090909 |
| Q6 | 0.50 | 70 | 14.285714 |

The period is `1 / flux_per_unit` A. The table rounds repeating decimals.
Microwave calibration multiplies full charge n in GHz per normalized DAC gain.
It does not normalize by n01. This environment has no fast-flux pulse calibration
or filter. The single-fluxonium [source-lines guide](../single_fluxonium/field-guide.md#source-lines)
describes the inherited bath approximations.

## Independent readout models

The backend uses six independent single-cavity models rather than one six-cavity
ADC model. Each mux tone and each pulse scan point routes by frequency to the
nearest bare cavity. An equal-distance tie raises an error. A 6900 to 7600 MHz
scan with 300 points avoids the five cavity midpoints in this configuration.
Sharing pulse ADC 6 does not combine all six cavity responses at one frequency.
Do not use this environment to infer inter-cavity radiative damping or collective
readout effects. Each qubit evolves independently, and each shot starts from
its configured equilibrium rather than the previous shot's final state.

Readout input uses sqrt(photons/ns) per gain. Receiver gain maps sqrt(photons/ns)
to ADC units. Offset is an I/Q pair, and the two-sided noise PSD is a 2-by-2 I/Q
covariance in ADC units squared times ns. Receiver phase uses degrees;
resonator mismatch phase uses radians. Time of flight uses ns.
Program input durations and trigger offsets use us, and frequencies use MHz.

Only terminal readout is supported. The model does not include cavity memory,
Stark shifts or backaction from an earlier readout pulse. At terminal readout,
the model uses DC energy levels and dressed cavity shifts. Weak dispersive
coupling and Markov baths remain assumptions.

Basis truncation and solver tolerances have different roles. A tighter solver
tolerance does not establish basis convergence. Check convergence again after
changing flux range, circuit parameters, drive strength or higher-state observables.
The six circuits differ, so convergence evidence for Q1 alone does not certify Q2
through Q6. Each uses native cutoff 40 and 21 retained levels as starting values,
not as a convergence claim. The solver uses rtol 1e-8, atol 1e-10, maximum step
0.1 ns and bath segments of 2 ns. Seeds 123 through 128 separate the six noise
streams. Startup performs no numerical warmup.

## Acceptance timings

The 2026-10-23 offscreen walkthrough loaded this environment's `startup.py`
into the measure GUI with the production experiment catalog and backend.
The window showed `six-fluxonium-demo / initial` and `Startup complete`.
The device list contained only the connected `flux_bias` FakeDevice.
No hardware connection or remote-control socket was opened.
A separate call to the public launcher `main(argv)` loaded the same script,
showed the window, then closed through GUI shutdown and returned exit code 0.

The walkthrough prepared inputs through the public Form resources, then clicked
Run with `ComponentDriver`. It selected each mux Run figure through its visible
picker and saved both the GUI screenshot and the plotted curve.
This checks startup and real adapter execution. It does not check every manual
form-editing action, Analyze, Writeback or Save.

All four Runs below ran in one process on heavy-slot CPUs 4 and 5 on an Intel
Core Ultra 7 258V, with BLAS limited to one thread and a 4,000,000,000-byte
address-space limit. Timing starts before the Run click and ends when Run is
available again with a visible curve. It includes compilation, acquisition and
Qt presentation, but excludes input preparation, screenshots, analysis and saving.
These are workload examples, not a speed comparison between experiments.

| Experiment | Requested sweep | Reps / rounds | Run wall time, s |
| --- | --- | ---: | ---: |
| `onetone/range_find` | 6900 to 7600 MHz, 300 points | 64 / 1 | 5.358448 |
| `mux/onetone_fine` | Common offset -3 to 3 MHz, 41 points, all six tones | 64 / 1 | 12.762754 |
| `mux/twotone_spectrum` | Per-qubit starting estimate +/-5 MHz, 41 joint points, Q1 through Q6 | 16384 / 1 | 4.051492 |
| `mux/t1` | 0 to 100 us, 31 joint delays, Q1 through Q6 | 16384 / 1 | 11.602649 |

The shared current stayed at 0 A. Every Run used 1 us recovery, a 2 us readout
pulse, a 0.9 us ADC window and a 1 us trigger offset. The pulse-route survey
used gain 0.05. Mux readout used gain 0.1 per tone and mixer frequency 7200 MHz.
The fine scan started from the template's bare cavity frequencies. Subsequent
qubit Runs used the acquired fine-scan minima as readout frequencies.

Both qubit experiments used 0.5 us constant drive pulses. Circuit eigenvalues
and charge matrix elements supplied starting frequency and amplitude estimates,
not measured calibration. The frequency sweep then acquired a response before
its strongest complex-IQ feature supplied the T1 drive frequency. Exact working
frequencies and gains belong to the walkthrough record, not the entry template.
No pulse library or qubit calibration was written back.

### Observed signals and limits

The range survey showed six separated dips near the six cavity bands. Their
sampled magnitude prominences were 18.81, 4.80, 12.70, 25.51, 7.80 and 6.43 a.u.
A 1 a.u. prominence threshold separated those dips from small baseline noise.
The six fine-scan figures each showed a dip, with magnitude spans of 72.14 to
85.64 a.u. Every acquired array was finite.

Q1 and Q6 supplied the qubit-response acceptance evidence. In spectroscopy,
the central five-point mean differed from the combined first and last eight
points by -0.909 and +5.604 a.u. The corresponding edge standard deviations
were 0.0605 and 0.1296 a.u. The GUI showed a central dip for Q1 and a peak for
Q6. In the T1 Run, the last five-point mean minus the first five-point mean
was +0.473 and -3.836 a.u. Their magnitude-versus-delay Spearman correlations
were +0.9891 and -0.9964. A relaxation response can rise or fall in magnitude;
the readout transfer function determines its sign.

The initial 64-rep qubit pilot had too much shot scatter for this judgment.
A nonzero curve span alone did not establish a qubit signal. At 16384 reps,
Q4's magnitude response remained small and non-monotone, and Q5's T1 magnitude
trace did not resolve a relaxation trend. This walkthrough does not certify
six calibrated transitions, pi-pulse fidelity, fitted T1 values or basis
convergence. The 100 us delay window need not reach the equilibrium baseline.
QICK emitted potential ADC crosstalk warnings for some tone spacings. Those
warnings do not establish physical crosstalk in these independent models.

### Memory and acquire-only evidence

A separate production-default six-tone acquisition used native cutoff 40 and
21 retained levels for every qubit, 64 reps, one round and 1 us pulse and ADC
windows. It reached peak RSS 436424 KiB, or 446898176 bytes. Cold acquisition
took 2.862611 s and the next warm acquisition took 0.202839 s. Both include
0.2 s pacing and ran consecutively in one heavy-slot call on CPUs 6 and 7.
Startup performed no warmup. These numbers describe acquire-only work and do
not replace the GUI Run times above.

The four-Run GUI process reached peak RSS 1195284 KiB, or 1223970816 bytes.
Both observed peaks were below 4 GB. Neither figure guarantees the same memory
use after increasing point counts, truncations or retained GUI figures.
