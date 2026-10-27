# Six-fluxonium settings guide

**Last updated:** 2026-10-27 (UTC), routing-only residency allowance.

**Readers:** operator, lab-author, framework-dev, agent. Read physical units, per-qubit parameters and approximation limits before changing this environment.

## Files and components

The tracked `default.yaml` template uses `zcu.six-fluxonium` version 1.
Copy it to the untracked `six_fluxonium.yaml` beside the startup script and edit
that working copy. Startup reads `six_fluxonium.yaml` and fails if it is missing;
the [startup instructions](README.md#trusted-startup) give the first-use copy command.
`entry-template.yaml` uses `zcu.entry-template` version 1.0.
The template creates Q1 through Q6, R1 through R6, local current sources
`Q1_bias` through `Q6_bias`, shared `coil_bias` and group `MG123456`.
Each qubit references its own local source. The group references `coil_bias`.
Scalar flux forms read `$qubit.flux_dev` on Open/Reset; no device alias is needed.
Tone i follows the group's
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
limits, initial current and ramp step. Each qubit's `dc_bias` contains its local
`source` settings and `offset`, `period`, `coil_offset`, `coil_period` in A.
Each nested Q1 through Q6 value keeps `format`, `version`, `chip`, `drive_line`,
`readout` and `numerics`. Its readout generator is 0 and its ADC is its mux tone
index. The schema requires source names `coil_bias` and matching `Qn_bias`.
It rejects zero periods, nonfinite calibrations and `flux_line` sections.
Old `offset_phi0`/`flux_per_unit` six-system documents have no schema fallback. Generators 7 through 12 are not configured.
Each resonator component instead declares
pulse wiring `ch: 13`, `ro_ch: 6`. `onetone/range_find` uses that pulse route.

## Receiver background

Each `qubits.Q1..Q6.readout.adc` receiver accepts its own `background`.
The shared pulse ADC6 uses Q1's receiver background, as it does Q1's gain and noise.

The optional receiver `background` object multiplies each carrier's microwave
I+iQ line response before receiver gain and phase. It uses the absolute incident
frequency, not the ADC demodulation frequency. Omission or an empty object gives
a flat unity background. Null is invalid.

| Field under `background` | Meaning and unit | Constraint and omitted value |
| --- | --- | --- |
| `ripple_depth` | Reflected standing-wave amplitude, dimensionless | Finite, 0 ≤ depth < 1; default 0 |
| `ripple_period_mhz` | Standing-wave period, MHz | Finite and positive; default 1000 |
| `ripple_phase_deg` | Standing-wave phase, degrees | Finite signed value; default 0 |
| `slope_db_per_ghz` | Amplitude slope, dB/GHz | Finite signed value; default 0 |
| `pivot_ghz` | Absolute frequency where the slope factor is unity, GHz | Finite signed value; default 0 |

The complex ripple changes amplitude and phase. The full background need not be
unity at the pivot. It stays fixed across rounds and does not change noise,
offset or intracavity physics. See the [ADC formula](../../../lib/zcu_tools/simulate/adc/README.md).

## Low-frequency drive evolution

Single-tone plateaus below 30 MHz use full-Hamiltonian midpoint matrix
exponentials with Strang bath half-steps, not the T6 Fourier channel builder.
Set top-level `numerics.slow_envelope_segments_per_period` to a positive integer,
default 20. It applies to all six sources and overrides the nested single-qubit
value. Increase it to check grid convergence. The accepted dense-reference
population error is 1e-2; Q1's 21-level, 50 ns, gain-0.2 calibration measured
about 1e-4 or less, and synthetic regression tests require 1e-3. These numbers
do not certify other gains, durations, coherences or basis truncations.
At 30 MHz and above, dispatch keeps the existing RWA/Floquet/Dyson/T6 order.

## Cache residency budgets

Set `numerics.routing_cache_bytes` in the top-level `six_fluxonium.yaml`
section, not in `qubits.Q1..Q6.numerics`. The default is 4194304 bytes, 4 MiB
across six sources. Each Engine receives one sixth, rounded down.
The value is a nonnegative integer. Zero disables retention.
At most five bytes remain unused. The cap excludes models, complete
preparations, numerical scratch and GUI/native allocations. It is not an RSS cap.

Engine prepare residency, exact-result memoization and source ADC solver
residency are removed. Existing working YAML files must remove
`prepare_cache_bytes`, `result_memo_bytes` and `adc_cache_bytes` before restart.
The loader does not silently accept those retired fields.
Single-source structure and complete preparations retain their own
[lifetimes](../../../lib/zcu_tools/experiment/single_fluxonium/README.md#reuse-and-lifetime).
The six sources remain independent and do not share receiver calibrations.

The measurements below describe the earlier prepare/result/ADC budget design,
not the current routing-only design. Their numeric results and historical
conclusions are unchanged. They do not measure the effect of removing those
layers. See the current [evolution contract](../../../lib/zcu_tools/simulate/evolution/README.md#evolve-local-preparations-and-caller-reuse).

## Cache budget measurements

The 2026-10-23 check ran `range_find`, `mux/onetone`,
`mux/twotone_spectrum` and `mux/amp_rabi` in that order. The check analyzed each
Run. It closed range_find after fine analysis, fine after spectroscopy
analysis, and spectroscopy then Rabi after Rabi analysis. Backend close followed
all tab closes. All work used the offscreen GUI, heavy-slot CPUs 2 and 3, and
one BLAS, OpenMP and Numba thread. The check used no hardware or measured data.

The survey used 6900 to 7600 MHz, 300 points and 64 reps. Fine used a common
-4 to 4 MHz offset with 81 points and one rep. Spectroscopy used six independent
60 MHz spans with 61 joint points. Rabi used 41 joint gain points. Both qubit
Runs used 65536 reps and one round. All four used a 2.2 us readout pulse and
1.12 us trigger offset. Mux Runs used a 0.99 us ADC window and mixer 7250 MHz.
Drive pulses were 0.1 us, except Q3's 0.2 us Rabi pulse.

The first old/new pair reduced the total prepare allowance from 1536 MiB to
256 MiB. It retained the same tracemalloc and ownership profiler in both GUI
processes. Old was `f5362637e`, before shared budgets; new was
`2eaf50e62`, with shared budgets. They ran consecutively in one heavy-slot call,
with separate initially empty Numba caches. A separate check used the reconciled
candidate `27757d875`, including main `bf67ce263`, without tracemalloc or ownership
scans. Its RPC handshake reported wire version 121. Every GUI exited with code 0.

| GUI measurement | Largest sampled VmRSS, bytes | Kernel VmHWM, bytes |
| --- | ---: | ---: |
| Old, with ownership profiler | 3239047168 | 3239047168 |
| 256 MiB prepare total, with ownership profiler | 2731687936 | 2731724800 |
| Reconciled candidate, /proc only | 1553829888 | 1559523328 |

These are GUI Python process measurements, not launcher RSS or marker maxima.
The sampler read `/proc` about every 20 ms; VmHWM also retains peaks between
samples. The paired profiled peak fell 15.66%. It still exceeded the original
1.5 GiB target of 1610612736 bytes. The final /proc-only peak was 51089408 bytes
below that target. This is one workflow observation, not an RSS guarantee.

At all 31 profiled markers with the 256 MiB prepare total, each source's cache
residency stayed within its share. The largest observed totals were:

| Cache | Divided six-source limit, bytes | Largest observed charged or resident bytes |
| --- | ---: | ---: |
| Prepare, including estimated metadata | 268435452 | 219742544 |
| Result memo | 67108860 | 7153274 |
| Routing | 4194300 | 1431488 |
| ADC, six separate allowances | 50331648 | 50149140 |

The largest simultaneous cache total was 278476446 bytes. After backend close,
all four cache totals and tab-owned arrays were zero. The profiled new process
still had RSS 2382536704 bytes and traced live allocations of 167072328 bytes.
Cache eviction and close release ownership; they do not force native allocators
to return pages. Active solver products, tracing bookkeeping, Qt and native
allocations are outside the cache counters. These measurements do not identify
which of those accounts for the remaining RSS. They do not diagnose a Qt leak.

The /proc-only check also uses a later main and omits profiling callbacks,
including forced collection at close markers. Subtracting its peak from the
profiled peak would not isolate tracemalloc's cost. Neither check measured native
heap allocations or the maximum active solver scratch separately.

The paired profiled Runs and analyses produced identical numeric field dtypes,
shapes and C-order bytes at all eight completion markers. This comparison used
raw numeric arrays and SHA256 hashes, not timestamp-filtered metadata.

### 768 MiB prepare recheck

The 2026-10-23 recheck used base `5d0ae6237` with
`prepare_cache_bytes: 805306368`. The same four-Run harness repeated the
settings, Run/analysis order and tab closes described above. Its configuration
edits and eight completion statuses matched the archived 256 MiB run.
The GUI loaded both the framework and demo startup from the changed checkout,
using that checkout's interpreter. It ran offscreen on CPUs 2 and 3 in a heavy
slot, with one thread for BLAS, OpenMP and Numba. It used no hardware or measured
data. All four Runs and analyses finished, and the GUI exited with code 0.

The observer sampled the GUI Python process's `/proc` status about every 20 ms
and saved numeric outputs at the eight completion markers. It did not enable
tracemalloc, recursive ownership scans or forced collection at close markers.
At startup and every completion marker, all six Engines reported prepare
limits of 134217728 bytes, or 128 MiB each.

| GUI measurement | Largest sampled VmRSS, bytes | Kernel VmHWM, bytes |
| --- | ---: | ---: |
| 768 MiB prepare total, /proc plus numeric dumps | 1868263424 | 1882968064 |

The kernel peak exceeds the earlier 1.5 GiB reference target by 272355328 bytes.
This run did not reproduce the earlier below-target result.
It is one workflow observation, not an RSS guarantee. The observer differs from both the
ownership-profiled pair and the earlier /proc-only run, so their RSS differences
do not isolate the prepare-budget change or identify native/Qt allocations.

All 733 numeric fields across the eight completion snapshots matched the
256 MiB run's manifests in dtype, shape, byte count and C-order SHA256.
The snapshots contain 115642 numeric bytes in total, including repeated fields
from tabs retained across markers. The 256 MiB run's raw files were no longer
retained. Its manifests matched the retained pre-shared-budget raw arrays.
The recheck compared its saved arrays directly with those retained arrays and
revalidated their manifest hashes. Every C-order byte matched.

### Paired warm Run timings

Both pairs below used six-tone Rabi with gain 0 to 0.5 at 41 points and T1 with
0 to 200 us delay at 30 points. Both used 200 reps, one round and 0.1 us constant
drive pulses at 0 A. Public model facts supplied starting frequencies; the cold
Rabi result supplied T1 gains. Each experiment ran cold then warm in one process.
Wall time includes cProfile and production pacing. Each old/new pair ran in a
single heavy-slot call on CPUs 2 and 3 with the same thread limits.

| Versions, old → new | Warm Run | Old wall, s | New wall, s | New/old |
| --- | --- | ---: | ---: | ---: |
| `f5362637e` → `2eaf50e62` | Rabi | 8.878339 | 8.879238 | 1.000101 |
| `f5362637e` → `2eaf50e62` | T1 | 6.500948 | 6.509018 | 1.001241 |
| `bf67ce263` → `27757d875` | Rabi | 8.892921 | 8.891226 | 0.999809 |
| `bf67ce263` → `27757d875` | T1 | 6.514958 | 6.512932 | 0.999689 |

Every ratio was below 1.10. Do not compare timings across the two pairs to
attribute main-branch changes to the cache budget. Each version started Rabi
with an empty disk cache. The first pair created 59 Numba cache artifacts per
version; the final pair created 57. T1 inherited only its own version's Rabi
artifacts. Warm Runs compiled or loaded no Numba functions. Each Rabi Run made
41 program admissions and rounds; each T1 Run made 30. The final pair's cold
and warm numeric arrays matched in dtype, shape and bytes, and its workload
inputs matched. All preparation and Run processes exited with code 0.

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

Each qubit's `chip.qubit` block also holds the loss and noise inputs below.
The values are in `default.yaml`.

| Field | Meaning | Unit |
| --- | --- | --- |
| `q_cap` | Constant Q of capacitive (dielectric) loss | Dimensionless |
| `q_ind` | Constant Q of inductive loss | Dimensionless |
| `xqp` | Quasiparticle fraction | Dimensionless |
| `aphi` | 1/f flux-noise amplitude of this loop | Phi0/sqrt(Hz) |

`temperature_k`, next to `chip.qubit`, is the bath temperature in K.
Each resonator uses external linewidth 1.0 MHz and internal linewidth 0.1 MHz,
both kappa divided by 2*pi. The full-charge coupling coefficient is 0.03 GHz.
There is no qubit-qubit coupling or crosstalk matrix. All six circuits respond
to the same live DC current.

## DC and control units

The seven DC FakeDevices `Q1_bias` through `Q6_bias` and `coil_bias` use A.
Each has limits -0.02 to 0.02 A, initial current 0 A and ramp step 1e-6 A.
Startup applies each initial current once. Each round captures all seven live
currents, retaining those raw A values in its detached snapshot.

For qubit i, reduced flux equals
`(I_coil - coil_offset_i)/coil_period_i + (I_local_i - offset_i)/period_i`.
Changing `Qn_bias` moves only Qn's operating point. Changing `coil_bias` moves
all six. The two contributions add, with no local-line crosstalk. Each offset
is a zero-flux current. Each period is a signed nonzero current per Phi0.
Negative periods reverse flux direction.

| Qubit | Local offset, mA | Local period, mA | Coil offset, mA | Coil period, mA | Flux at zero currents, Phi0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | -1.0 | 2.0 | 0 | 10 | 0.50 |
| Q2 | -1.125 | 2.5 | 0 | 12.5 | 0.45 |
| Q3 | -0.99 | 1.8 | 0 | 8.333333 | 0.55 |
| Q4 | -0.88 | 2.2 | 0 | 11.111111 | 0.40 |
| Q5 | -1.68 | 2.8 | 0 | 9.090909 | 0.60 |
| Q6 | -1.5 | 3.0 | 0 | 14.285714 | 0.50 |

The table rounds repeating decimals. YAML stores these coefficients in A.
For Q1, local current 0.1 mA and coil current 1 mA give reduced flux 0.65.
Each single-circuit backend receives the summed fixed flux. It does not read
other qubits' local lines. Calibrations written to an entry do not change these
physical settings.
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

The entry template sets group `mixer_freq: 7250` MHz for readout centers from
7000 to 7500 MHz. Blank mux forms read `$mux.mixer_freq`; missing or null stays
required and blank, without deriving a midpoint or depending on kappa.
Group `time_of_fly: 0.4` us matches the receivers' `time_of_flight_ns: 400`.
Resonator pulse routes have separate top-level `time_of_fly: 0.4` us values
from their own receivers. The template seeds total linewidth 1.1 MHz and both
local-line and coil calibrations as simulated-truth, not measurement results.
Each line's calibration uses `int=offset`, `half=offset+period/2` and its own
period in A; coil records use `coil_offset` and `coil_period`. These describe
each line's contribution alone, excluding the other line's background flux.
For blank `mux/onetone`, trigger starts at `$mux.time_of_fly + 5/(2π·min κ)`,
ro_length at 1.0 us, and length at trigger plus ro_length. Missing group TOF or
any kappa leaves trigger and length required and blank. These editable seeds
are not device-confirmed calibration. Readout slots retain their mixer and
existing timing rules.

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
0.1 ns and bath segments of 2 ns. Q1's seed (123 by default) selects the
composed environment's root stream. The backend derives distinct cavity streams
from that root. Q2 through Q6 seeds do not control their corresponding noise
streams. Startup performs no numerical warmup.

## Acceptance timings

The 2026-10-23 offscreen walkthrough loaded this environment's `startup.py`
into the measure GUI with the production experiment catalog and backend.
The window showed `six-fluxonium-demo / initial` and `Startup complete`.
This earlier acceptance run used the former single-global-line topology.
Its device list contained only the connected `flux_bias` FakeDevice. The current
topology installs seven devices; the timings below are not a seven-device recheck.
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
| `mux/onetone` | Common offset -3 to 3 MHz, 41 points, all six tones | 64 / 1 | 12.762754 |
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
