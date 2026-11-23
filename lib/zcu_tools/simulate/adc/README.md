# ADC probability and readout contracts

**Last updated:** 2026-11-23, simulator optimization index

**讀者：** framework-dev、agent。ADC 機率、讀出訊號與 interval moment 契約。

[ADR-0071](../../../../docs/adr/0071-coherence-units.md) defines frequency and decay-rate units.
[ADR-0072](../../../../docs/adr/0072-module-boundaries.md) places ADC numerics in `simulate`, outside model environment assembly.
[ADR-0078](../../../../docs/adr/0078-evolution-module-boundary.md) assigns field integrals, bounded field caching and complete raw sampling to ADC.

See the [simulator optimization index](../../../../docs/simulator-attempts/INDEX.md) for attempts and historical measurements.

Importing `simulate.adc` initializes the shared simulator Numba cache policy
before creating dispatchers, even if evolution has not been imported. ADC and
evolution share one successful bootstrap per process. Basic coordinate
conversions at the `simulate` root do not initialize it. See the
[cache contract](../README.md#numba-machine-code-cache) for source/version
isolation, fail-fast storage and process-global effects. Raw-path dispatchers
with `cache=False` still compile in each process.

<a id="readout-model"></a>

`ReadoutModel` supports 1 or 2 cavities on a shared transmission line.
`cavity_count` comes from the frequency array's second axis. Other counts raise
`ValueError` at model construction. Frequencies have shape `(state,cavity)`;
internal/external decay rates and scattering phases have shape `(cavity,)`.
The two-cavity default `mismatch_phase_rad=zeros(2)` is unchanged. One-cavity
callers must supply a one-element phase array, including `np.zeros(1)` for zero
mismatch. Receiver I/Q offset and noise covariance remain `(2,)` and `(2,2)`.
The scalar path propagator is exact; two-cavity paths retain the existing 2x2
closed form. Constant, sampled and opaque drives share these model dimensions.

`ReadoutModel.rates_per_ns` follows the model's state-label order, with
`[target, source]` entries in 1/ns. In raw and trace sampling,
`_build_event_plans` creates an observer for an ADC/content pair only when the
event has both incident pulses and observation grids for that pair. A
receiver-only window does not create an observer. For each content group in an
event, sampling builds one shared `JumpPaths` to draw continuous-time Markov
chain (CTMC) jumps from these rates if the group has at least one observer. A
group with no observer skips this step. Between-event
[projection transition probabilities](#projection-chain-sampling) are separate
from these within-event jumps.

FTF's
[`readout_model`](../../simulators/ftf/physics/README.md#readout-and-floquet-paths)
supplies the rates in retained DC energy-state order.

`cavity_fields` returns complex `(time,state,cavity)` fields in fixed joint-state
order, without population mixing or receiver calibration. Its optional complex `(cavity,)`
`initial_field` supplies the same field to every state at `initial_time_ns`.
`cavity_field_average` accepts the same pair of inputs. Both use sqrt(photons)
in the model's rotating reference at that absolute emitted time. The initial
field defaults to zero and the initial time to 0 ns. Omitting both equals
explicit vacuum at t=0, including mid-pulse queries. Propagation uses only later
pulse history without restarting pulse-local phase or envelope coordinates.
Queries before the initial time fail with both times in the error. `cavity_field_average`
returns `CavityFieldAverage`: photon occupations and same-time field products
averaged over an emitted-time interval, including coherent carriers and full
ringdown. Flat inputs use analytic constant-line propagation and quadratic
integrals. Shaped inputs retain the field ODE and converged quadrature, split at
pulse boundaries, envelope corners and the decay timescale. ADC owns these
field integrals; consumers project them into their own Stark or dephasing models.
See the [FTF cavity span handoff](../../simulators/ftf/environment/README.md#cavity-drives)
for how FTF supplies `initial_field` and `initial_time_ns` at span boundaries.

`CavityAverageCache.average` reuses flat second moments. `CavityAverageCache.span`
returns both second moments and final fields for all states in the original model
reference. It shares flat propagation when the initial time equals the span
start and both solvers use the same time boundaries. Its exact key keeps model
content, initial field bits, and pulse order, timing, carrier, phase and
breakpoints. It distinguishes numeric scalar types and bits without using their
display format. Pulse amplitudes starting at or after the span end do not
distinguish span keys. Shaped/callable pulses, an empty pulse tuple, earlier
history and inputs with unproven equivalence use separate solvers without
admission. The field solver must accept the endpoints as float64 query times;
rounding that makes them coincide raises `ValueError`.

FTF projective continuation and non-projective (MIST) evolution both call
`span` for cavity drives. `average` and `span` share one byte budget, 64 MiB by
default. Results are writable copies independent of resident arrays. Full
capacity declines new entries without eviction. Callers serialize access and
clear the owner at the end of its lifetime. `cavity_field_average` and
`cavity_fields` remain uncached.

ADC consumes explicitly ordered classical populations, not densities. The
probability tolerance is `POPULATION_TOLERANCE = 1e-6`. This is 1000 times below
the segment population budget of 1e-3 and gives about 38 times margin above the
measured shaped-kernel trace drift of 2.62e-8.

Environment authors explicitly call `normalize_populations` after converting
densities into readout-basis populations. This is the only probability repair
operation. It checks finite values, a nonempty state axis, row sums in the closed
float64 interval [1-1e-6, 1+1e-6]. It then clips every negative entry to zero
and divides each row by its sum, returning an owned array without changing its
input. There is no lower bound on negative entries. It never clips upper values
or accepts row sums outside tolerance. Invalid shapes raise `ValueError`.
Row validation raises `PopulationValidationError`, a `ValueError` subclass whose `rejected_rows: int`
counts rows failing the first reported rule. It returns no partial results.
Callers can retain this count without parsing error text.

`PopulationInput`, `ProjectionReadout` and `ProjectionChain` validate the same
probability trace interval and require nonnegative entries. `round_statistics`,
`PreparedStatistics`, `PreparedRawShots` and `sample_traces` take either
`ProjectionReadout` or `ProjectionChain` as their first argument.
They expect explicitly normalized distributions and do not normalize or clip them. Their tolerance accommodates
numeric residuals; it does not replace explicit preparation. State ordering must
match the readout model and conditional responses. No basis inference occurs in
ADC. Evolution retains its raw densities and reports trace drift separately.

`ConditionalReadout.response` and `responses` return microwave I+iQ. They
conjugate the demodulated physical field before receiver gain, phase, offset and
noise. Receiver calibration does not conjugate or rotate the added noise's I/Q
covariance. Constant and shaped pulses use the same convention.

A `SampleGrid` carries an explicit demodulation frequency `f_demod`. Its
oscillator uses the signed detuning `f_demod − f_ref`, multiplied by absolute
shot time. The demodulation phase is specified at shot time zero. Multi-tone
models share `f_ref` across tones; each grid uses its own `f_demod`.

`Receiver.background(f)` supplies a fixed complex transmission background at
each incident carrier's absolute frequency `f` in GHz:

```text
B(f) = 10**(slope_db_per_ghz * (f - pivot_ghz) / 20)
       * (1 + ripple_depth * exp(i * (2*pi*f/ripple_period_ghz + ripple_phase_rad)))
```

The ADC multiplies each carrier's microwave I+iQ line response by `B(f)` before
receiver gain and phase, then adds offset and noise once after superposition.
It never evaluates the background at the ADC demodulation or rotating-reference
frequency. The slope factor is unity at `pivot_ghz`; the complete background
need not be unity there when ripple is nonzero. Zero depth and slope give unity
at every frequency. Depth is in `[0,1)`, period is positive, and all five
parameters must be finite. Optional extensions multiply this legacy background:

- `extra_ripples` is a sequence of `BackgroundRipple(depth, period_mhz, phase_deg)`.
  Each contributes `1 + depth*exp(i*(2*pi*f*1000/period_mhz + phase_deg*pi/180))`.
  Depth must be finite in `[0,1)`, period finite and positive, and phase finite.
- `roughness=BackgroundRoughness(amplitude, correlation_mhz, seed=0)` contributes
  `1 + z(f)`. Amplitude is the finite nonnegative complex RMS `sqrt(E[abs(z)**2])`,
  not the RMS of each quadrature or of `abs(B)`. Correlation is a finite positive
  frequency separation in MHz where the ensemble correlation falls to `e^-1`.
  The field approximates `exp(-(df/correlation_mhz)**2)` correlation with 512
  Gaussian-distributed spectral modes and independent uniform phases.
  Finite-band RMS and correlation fluctuate around these ensemble definitions.
  A local PCG64 stream uses `SeedSequence(seed, spawn_key=(0x424743,))`.
  The nonnegative integer `roughness.seed` fixes the line shape independently
  of `numerics.seed`, acquire identities and receiver noise. Equal settings,
  seed and absolute frequency return identical values regardless of scan order,
  spacing or point count. Global RNG state is unchanged. Reproduction assumes
  matching NumPy and numeric implementation versions.
- `band_edges=BackgroundBandEdges(low=..., high=...)` accepts either or both
  `BackgroundBandEdge(corner_ghz, rolloff_db_per_ghz)` values. Corners must be
  finite, rolloffs finite and positive, and the low corner below the high corner.
  Outside each corner the attenuation grows linearly in dB with frequency
  distance. Inside the band it is zero. Amplitude is continuous at each corner;
  its derivative may change there. Omitted sides do not attenuate.

The extensions default to an empty tuple, `None` and `None`. Omitting them
preserves the legacy result point for point. Zero roughness amplitude disables
its perturbation. In YAML the same nested fields live under each receiver's
`background`; `extra_ripples` uses a list of records with `depth`, `period_mhz`
and optional `phase_deg`. FTF and single-fluxonium use the shared schema;
six-fluxonium delegates each cavity to single-fluxonium. Format/version do not
change. Their tracked `default.yaml` templates enable 68/44/308/52.6 MHz waves,
2% complex RMS roughness, a -4 dB/GHz slope and a 3.7 GHz low corner. Existing
untracked working copies do not change. Add these fields to your working YAML
and restart the environment to see the new background.

Raw paths, conditional traces and integrated sums
share this convention, including shaped and simultaneous carriers. Linear
propagation permits weighting private ADC carrier amplitudes by `conj(B(f))`
before field superposition and final conjugation. This does not change physical
`cavity_fields`, photon averages, Stark shifts or backaction. Both readout and
flat cavity-average cache keys include the legacy parameters and every optional
background record, including the fixed roughness seed.

`Receiver.time_of_flight_ns` is one finite nonnegative total line delay, default
zero. Call `adc_to_device_time_ns(t, time_of_flight_ns=receiver.time_of_flight_ns)`
to convert ADC times in ns to device times t−τ. Use the result once, for emitted
time, and keep the original grid times for demodulation. Negative device times
are valid. The solver evaluates the physical output at that device time and restores
the rotating frame's exp(+2πi*f_ref*τ) before demodulation at ADC time t. Conjugation then gives
exp(−2πi*f_abs*τ) at the absolute carrier, without counting τ twice. Onset,
ringup, ringdown and envelopes all arrive late; offset/noise do not. Pulse phase
retains its documented rotating-coordinate convention.

`ReadoutModel.mismatch_phase_rad` is a finite array with one phase per cavity.
Each element multiplies only that resonator's scattered field by exp(−i*phi)
before conjugation, giving the hanger model's +phi sign in microwave IQ. The
reciprocal shared-line generator and transition rates are unchanged. This is a
scattering-phase model, not a full impedance network or a global signal rotation.

`ReadoutEvent(pulses, grids)` groups coherent incident pulses and ADC windows
that share one physical read event. `ReadoutBatch(model, events, content_index)`
stores ordered events per distinct content. Every grid in an event observes the
sum of its incident carriers through the same physical line. Receiver offset
is applied once per sample, not once per tone. Responses concatenate grids in
event order. `ReadoutEvent.pulses=()` keeps observed grids and emits receiver
offset and noise only, without CTMC or cavity propagation. `ReadoutEvent.grids=()` keeps an event's position when that ADC
does not observe it. It emits no read column. All ADCs and projection populations
use the same event axis, while each ADC keeps its own observed trigger order.
All ADCs may omit terminal observations. The terminal event still occupies the
last position in the common axis and contributes its populations to integrated
counts, but produces no terminal read column. Each ADC content still requires
at least one observed grid across its events.
`ReadoutBatch.output_order` optionally maps each output column to an event-major
column, separately for every content. `content_index` selects each point's row.
None means identity order. Explicit rows must be complete int64 permutations.
`ConditionalSums` retains event-major sums, traces and `event_indices`, and carries
the same output permutation. Integrated sampling uses each read's logical
event to select conditional means in output-column order. Raw and decimated
sampling reorder after the shared paths and receiver noise.
This preserves chronological physics while restoring the original channel-local
trigger emission order, even when it differs between points.
`ConditionalReadout` computes state-conditioned responses once
per distinct content. Integrated flat (`envelope=None`), common-start mux
contents sum each carrier's linear response through the same line. Different
pulse end times retain their own ringdown. Each carrier first tries guarded
direct discrete sums on uniform ADC grids. A rejected carrier uses the existing
constant-field solution, including full block exponentials when the coupling
guard rejects the projected solution. Guards are not relaxed and one rejection
does not send other carriers to the ODE. Calibration includes receiver offset
only once per sample after superposition. The workspace stays local to the
`ConditionalReadout` instance, not shared globally or across receivers.
Flat common-start trace/decimated contents also superpose vacuum-start carrier
fields through the same `ConstantLine`. Homogeneous propagators and sampled
scattering projections are shared across carriers, with the existing full block
exponential fallback for ill-conditioned coupling. Calibration adds receiver
offset once after field superposition. Different-start multi-pulse contents and opaque envelope callbacks retain the
field ODE. `PiecewiseLinearEnvelope(knots_ns, values)` owns read-only copies of
finite complex values and increasing pulse-local ns starting at zero. It linearly
interpolates and holds the final value through pulse end. Its lossless content
key allows independently constructed copies to reuse responses. Common-start
contents containing these envelopes superpose carrier-frame affine-input
solutions. An augmented exponential supplies population/moment propagators and
constant/linear forcing convolutions, including singular/defective generators.
The workspace retains only the current knot state, ADC outputs, and at most 128
step blocks per carrier. Nonfinite blocks retain the ODE fallback. Flat carriers
in mixed sampled/flat contents still use ConstantLine. Calibration occurs once
after superposition. The public single-grid `response` remains an ODE reference.

Integrated flat single-carrier contents sharing exact start/end and per-read
time coordinates use tiles of at most 64 carriers. Amplitude, carrier phase and
demodulation may differ. Uniform-grid validation and population evolution are
shared; separation, cancellation and residual guards remain per carrier. Rejected
contents retain their scalar sum/field fallback. No content-by-sample dense
moment tensor is constructed.
Integrated single flat pulses also use the guarded direct discrete sums. The constant moment solution
comes from `(M+iω)X−XQ=B`. In the eigenbases of M and Q, its coefficients are
`Y_ab=(Um⁻¹ B Uq)_ab/(μ_a+iω−λ_b)`. Each demodulated exponential sums to
`exp(z*a)*expm1(N*z*h)/expm1(z*h)`; the sampled-alias limit is `N*exp(z*a)`.
The incident active segment, causal on-step and opposite off-step are summed
separately. The off-step retains `exp(Q*T)` population transfer. Conjugation,
receiver gain/phase/offset, delay and mismatch follow the field convention.
There is no sample-period factor, noise draw or population normalization here.

Both eigenbasis condition numbers must be below 1e4, reconstruction residuals
below 1e-13 and spectral separation at least 1e-10. The original Sylvester
cancellation and residual guards also apply. Uniformity compares sample
coordinates to the endpoint-fitted lattice, within four coordinate ULPs plus
eight float64 epsilons times the grid span. Unlike a fixed ns tolerance on
adjacent differences, this accounts for quantization at large absolute starts.
Single samples are uniform. Direct sums reuse the exact-content cache across
contents and batches. Trace requests retain a separate field cache.

Create `ConditionalReadout(model, rtol=..., atol=...)` for the current preparation
and call `responses(batch)`. Its internal exact-content reuse stays within that
solver's lifetime. Sources do not retain ADC solvers across preparations.
Caller-owned complete preparations may retain deterministic conditional sums.
No solver cache stores noisy IQ.

Unsupported direct sums fall back individually, including nonuniform grids
and defective generators. Single-pulse trace requests retain their original
field route. Opaque callbacks retain the ODE. Flat fields share frequency-independent Schur bases and sampled output
projections when enough contents reuse the same relative grid to amortize their
construction. These grids evaluate scattered fields directly without advancing
a dense moment tensor at every sample. Other grids retain step propagation with
shared Schur bases. Ringdown subtracts the delayed step response;
ill-conditioned driven blocks retain the full matrix-exponential propagation.
The field routes do not assume diagonalizable rate matrices. No route changes
the evolution populations.
Its lossless numeric cache key includes delay and every cavity's mismatch phase.
`ADC.sample` and `sample_windows` draw continuous-time Markov paths
using the directional rates. The initial population origin is the earliest
incident pulse start, or zero with no pulses.
The path continues through ringdown to the latest emitted sample, including gaps
between windows. Jumps retain every resonator's field and reciprocal shared-line
damping. Vacuum baselines support flat, piecewise-linear and opaque envelopes;
no ODE is solved per shot. Delay, mismatch and receiver calibration follow the
same conventions as conditional responses. Windows share one path per shot and
add independent receiver noise. Zero rates retain deterministic conditional
traces. Multiple incident carriers superpose their vacuum fields and retain
one cavity memory on the same path. Integrated direct means retain conditional
moments and nearest-even shot rounding variance. Acquisition averaging belongs to program.

## Projection readout and sampling

`ProjectionReadout(weights, populations)` owns outcome weights with shape
`(*point_dims, K)` and one `(*point_dims, K, states)` population array per event.
It validates and copies these arrays without clipping or normalization.
`ProjectionReadout.unprojected(populations)` explicitly constructs one outcome
and one event. A single outcome consumes no extra outcome RNG draw. Its raw,
trace and integrated outputs retain the previous single-event RNG order,
including shared-event multi-grid and multi-ADC arrangements.

<a id="projection-chain-sampling"></a>

`ProjectionChain(initial, transitions)` represents rank-1 projective reads
without enumerating outcome histories. `initial` has shape `(*point_dims, S)`.
Each subsequent event supplies one `(*point_dims, S, S)` transition array,
indexed `[previous_projection_state, next_projection_state]`. Construction
copies arrays read-only and validates every row without clipping or normalization.
An empty transition tuple represents one event.
[FTF projective chain construction](../../simulators/ftf/environment/README.md#projective-chain-construction)
builds these rows and applies the projection floor before handing them to ADC.
FTF passes a `ProjectionChain` for projective acquires and uses the single-event
`ProjectionReadout.unprojected` without projections. The general `ProjectionReadout`
outcome-mixture input remains available to ADC callers, but FTF does not use it
as a projective fallback. See [ADR-0078's Markov chain decision](../../../../docs/adr/0078-evolution-module-boundary.md#projective-read-的-markov-chain-表示).

Raw and trace sampling each draw a chain per shot. Reads at the same shot
index stay paired across events and ADCs. ADCs and grids that observe the same
event share its CTMC path. The next projection depends on the previous
projection state, not on the readout CTMC endpoint. Raw and trace paths are
sampled separately from compact flow counts. No compact bin identifies a shot.
Chain sampling uses its own draw order. Equal inputs and identities reproduce
results within the same numeric/RNG version; no bitwise parity with enumerated
histories is promised. The existing single-event `ProjectionReadout` RNG order
stays unchanged.

<a id="integrated-chain-statistics"></a>

Integrated chain sampling draws first-event occupancy with a multinomial.
For each next event and source state it draws multinomial destination flows,
then sums the incoming flows to obtain that event's histogram. No shot array
or outcome history is constructed. `cumulative_count_bins(histograms)` encodes
these histograms on sorted cumulative-count boundaries. It returns
`HistogramBins.counts` and a tuple of per-histogram category `indices`.
All histograms use the same point shape and total per point. Each total must
be in `[0, 2**63-1]`, inclusive; larger totals raise `ValueError` before boundary
encoding. Counts and indices remain int64, without clipping or normalization.
The shared width is `1+sum(S_e-1)`, independent of reps; duplicate boundaries
remain zero-count bins. Each read uses its `event_indices` entry and
content-specific column order to select means on these bins. An ADC that skips
a herald emits no column for it and keeps the later event indices. The terminal
event contributes to the bins even if no ADC emits a terminal column. The same
encoder can combine independent cavity histograms with different category counts.

For FTF projective reads, the bins are the shared `state_counts`, partitioning
reps at each point. They encode each event's marginal histogram, not a joint
distribution over herald history and terminal state. Each read's
`conditional_sums` holds its noiseless per-shot mean for each bin. Summing
`state_counts * conditional_sums` over bins gives the noiseless total for
each I/Q component. The common flow draw preserves the correlations between
these totals due to event occupancies. It does not yield the joint IQ
distribution of paired reads, CTMC trajectory variance, or cross-read
receiver-noise covariance. ADC adds Gaussian noise to each read total
independently, including the nearest-even rounding variance term. See [FTF compact statistics](../../simulators/ftf/environment/README.md#compact-statistics)
for the environment's output and the [Program statistics contract](../../program/v2/simulation/README.md#compact-timelines-and-statistics)
for the acquisition handoff.

The environment supplies projection probabilities in the retained DC energy
basis. ADC samples the supplied rows, including identity rows, without applying
another floor or choosing projection times. It does not group states by IQ
distinguishability. ADC transitions during a herald event affect that event's
IQ only. They do not feed back into the populations supplied for later events.

## Integrated raw shots

`PreparedRawShots(readout, batches, reps=...)` owns noiseless probabilities
and per-channel ReadoutBatch inputs. `bind(identity)` returns an `ADCRawShots`
source without doing shot work. Materialization draws full CTMC paths, one per
logical event. Within an event all carriers, ADCs and grids
share the path. Different events draw independent paths after selecting one
outcome shared by the whole shot, or following the chain's projection states.
Each receiver adds
independent noise. Zero jumps use direct vacuum sums. Jumped shots use compiled exact homogeneous propagation and discrete
sample sums. Uniform samples use binary geometric sums; irregular samples keep
their original coordinates. When a jump wave saves enough work to pay for setup,
ADC sums cavity memory using a difference of per-state projected propagator prefixes
at the actual sample times and one propagation per segment.
Flat baselines use singular-safe phi1 propagation.
Shaped and opaque envelopes retain the existing DOP853 vacuum-baseline tolerances.
There is no ODE per shot, jump-time LUT, path-count conditioning or tail cutoff.
Windows within one event share its path through ringdown and gaps. Offset is
added once per sample.
All ADC models must declare identical state labels and directional rates.
Different receivers may deduplicate contents differently; point mappings select
physical inputs independently without changing the common path.

An acquire that reuses a prepared batch calls
`RawPlanOwner.bind(prepared, identity)` instead of `prepared.bind(identity)`.
ADC compares prepared objects by identity and retains plans for only the current
batch. It builds each logical event's plan on demand during raw materialization.
[FTF](../../simulators/ftf/environment/README.md#preparation-reuse) and
[standalone single-fluxonium](../../simulators/single_fluxonium/environment/README.md#reuse-and-lifetime)
retire the plans when the batch changes or the acquire ends, without moving
plans into the source preparation cache.
After retirement, previously delivered sources recompute their plans without
filling either the old or the current memo. Direct `bind(identity)` and decimated
traces do not use an owner.

The source retains its inputs independently of later rounds and preparation
eviction. Every call restarts its private PCG64 stream and returns fresh int64
arrays, keyed by ADC channel with shape `(*loop_dims, reads, 2)`. It does not cache
noisy results or adjust shots to independently drawn mean totals. Invalid reps,
layout, shapes or states raise ValueError, as do unsafe integer sums. Solver
failures raise RuntimeError. `simulate.rounding` owns shared numeric int64
conversion and shot-axis layout. Numeric sources do not import program.

`materialize(reps, avg_level)` remains available without a Stop query.
`materialize_cancellable(reps, avg_level, cancel=...)` borrows a read-only query
for the current call. It checks at entry, at each logical event, after each
point group with shared ADC contents, and before returning. If it observes Stop,
it raises `simulate.boundary.Cancelled` without returning partial arrays.
The source does not retain the query. When Stop is not observed, both methods
use the same RNG stream and batch order and return identical shots.

`sample_traces(readout, batches, reps=..., identity=..., rtol=..., atol=...)`
requires physical `ReadoutBatch` inputs keyed by ADC channel in declaration order.
It returns `RoundSignals` with owned float64 arrays shaped
`(prod(point_dims)*reps, reads, samples, 2)`, in point-major C order then shot order.
Each ADC requires equal sample counts across reads and contents. The complete
`SamplingIdentity` seeds PCG64 without an extra suffix. FTF supplies
`(acquire_sequence, round_index, 0)` as its stream. Windows of one event share
the full CTMC path, including times not observed by any window. Delay, mismatch and demodulation
remain per receiver/grid; noise is independent per ADC/read/sample/shot. There
is no integrated rounding term. Invalid probabilities, states, rates, layouts,
counts or tolerances raise `ValueError`; baseline/field failures raise
`RuntimeError`. There is no conditional-only trace sampler or fallback.

`PreparedStatistics(readout, channels, reps=...)` validates and owns immutable
integrated sampling inputs. For `ProjectionReadout` it draws counts on the C-order flattened
`(outcome, terminal state)` axis with probabilities `weights[k] * final[k, s]`.
Earlier-event conditional sums average that event's populations for each outcome,
then repeat across terminal states. Terminal-event conditional sums depend on
the terminal state and repeat across outcomes. Every ADC/read uses the same
joint counts over `(outcome, terminal state)`. Preparation expands conditional
means and factors each ADC's noise covariance once.
`draw(seed=..., acquire_sequence=..., round_index=...)`
validates a fresh identity and uses the same count/noise algorithm as
`round_statistics`. `ProjectionChain` uses flow counts and the cumulative-boundary
encoding described above. The preparation retains no noisy results or identities.
The public `round_statistics` still validates arbitrary inputs by building a plan
for that call. FTF retains plans with noiseless preparations; no RNG draws or
deferred raw materialization move into preparation.

### Raw sampling

A change in representation alone must preserve the joint distribution of raw
I/Q across reads and ADCs at the same shot index, not just the distribution of
each read by itself. With the same physical model and inputs and no projection
floor approximation, sampling a `ProjectionChain` and sampling the enumeration
of those same paths are equal in distribution. This does not establish parity
with an FTF history model that retains cavity memory or uses a different floor.
The [chain's RNG boundary](#projection-chain-sampling) does not promise bitwise
parity for the same seed across representations. See ADR-0078 for
[raw ownership](../../../../docs/adr/0078-evolution-module-boundary.md#integrated-raw-由環境生成介面由-program-擁有)
and the [Markov representation](../../../../docs/adr/0078-evolution-module-boundary.md#projective-read-的-markov-chain-表示).

On the simulated ADC path, these callers use raw rather than compact counts:

- Shot reducers use `acquire_summed`, including through `run_acquire_summed`.
  In `twotone/ro_optimize`, this includes Auto (`auto_optimize`) and the
  `freq`, `freq_gain`, and `length` scans. In `jpa`, `auto_optimize`, `flux`,
  `freq`, and `power` use raw for SNR. `Classify` axes passed to `shot_sums`, as
  in `two_qubit/joint_readout` and heralded `conditional_phase`, classify the
  reducer's per-shot IQ.
- In `SimulationAcquireMixin`, a registered shot mask (including `HeraldReset`)
  or a native QICK `threshold` requires raw during an otherwise averaged
  acquire. Direct `SimulationSoc.start_readout` also requires raw for `poll_data`.
- `get_raw` and `get_shot_result` request the last completed round's raw.
  `run_acquire_shots` reaches it through `get_shot_result`; mux single-shot g/e
  reaches `get_raw` through `collect_raw_iq`. These getters only materialize
  the retained round if its raw is not already available.

See [Program's raw materialization contract](../../program/v2/simulation/README.md#raw-materialization)
for when a round requires raw and when getters materialize it later.

To locate the callers, use:

```bash
git grep -nE '\b(run_acquire_summed|acquire_summed|Classify|HeraldReset|register_shot_mask_source|SimulationAcquireMixin|start_readout|run_acquire_shots|get_shot_result|collect_raw_iq|get_raw)\b' -- '*.py' ':!tests/**'
```

## Joint-Fock validation limits

The independent [evolution joint-Fock prototype](../evolution/README.md#joint-fock-prototype)
requires an explicit request. It does not feed ADC or change Engine's flat or
shaped routes. The conservative two-state path uses analytic one-jump prefill,
a sampled-pointer cutoff with one extra excitation, and fixed CSR RK4 steps
no larger than 1 ns. Its selected population and qubit-line-centre comparisons
agree with the saved reference within their local statistical gates.
Those gates do not bound errors at other coordinates. Fitted cavity-centre
differences remain diagnostics, not acceptance criteria or accuracy bounds.

An explicit `JointFockDiagnostic` selects numeric physical-vacuum, finite-pi,
prefill and probe preparation with two or four retained levels. Its
caller-selected `step_ns` can permit local diagnostics where the default 1-ns
step produces nonfinite densities. Local two-level step convergence does not
validate a full map, and the four-level sample has no separate step-halving
study. Comparisons with the analytic path change both preparation and cutoff
selection, so they do not isolate dynamic-level truncation error.

The sampled coherent cutoff rule is not a complete-density error bound.
A generator norm times the step only bounds scaled eigenvalue magnitudes;
it is not an RK4 amplification factor or a stability or accuracy proof.
Fixed RK4 does not guarantee positivity for arbitrary inputs. The prototype
preserves finite population and density defects without clipping or repair.
Nonfinite evolution fails without an automatic step reduction, retry or
solver substitution. It remains opt-in, not a production acquisition backend.
