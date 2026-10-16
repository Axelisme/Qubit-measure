# FTF environment composition

**Last updated:** 2026-10-17

`ftf(path_or_settings)` builds the same `SimulationEnvironment` for GUI startup
and Notebook callers. It creates fixed-unit V/A `FakeDevice` sources in one
`DeviceManager`, applies their initial values once and binds the selected channel
counts to the fixed mock QICK profile. The returned manager owns device cleanup.
Builder failures close devices and preserve startup and cleanup errors.

`FTFSource.snapshot` reads live DC values before each round. It checks units and
limits without restoring startup values. `execute` consumes detached templates.
One source owns one `Engine`, whose prepare cache survives across acquires.
Workers exist only during `evolve`; the engine retains no idle threading pool.
`FTFSource.close()` clears preparations and the engine cache. Dropping the
environment also releases this in-memory state; no disk cache is involved.

`open_acquire(request)` starts a session for a fixed request. Its
`sample_round(snapshot, index)` reuses the preceding round's density and
conditional response when the snapshot is equal. Every round draws a new stream.
Opening another acquire invalidates the old session and drops its preparations.
The binding serializes calls; direct author callers must do the same.

`plan_template` converts QICK us/MHz/degrees into ns/GHz/radians with point
arrays. It preserves signed gains, complex sampled envelopes and the common
carrier origin. Pulse supports use absolute nearest-2-ns rounding, ties even.
The model owner supplies a DC `ModelSource` and filters flux tracks into a ragged
lattice schedule. The engine evolves every template together to readout start.
The current batch conversion keeps flux-track wiring in one function. Continuous
flux waveform transport is added separately with the continuous-flux kernel.

Readout uses the DC model and populations in its energy basis. It ignores flux
tails and controls during or after readout. Earlier control portions still evolve
normally. Each point must have exactly one active readout pulse, including when
several mutually exclusive slots represent that pulse. The environment supports
one bound ADC with multiple trigger windows. Constant readout envelope values
fold into amplitude with `envelope=None`, enabling the exact field solution.
Conditional responses are solved once per distinct content, not per shot.

The receiver noise PSD is two-sided, in ADC sample units squared times ns.
Discrete sample covariance is PSD/dt_ns using the ADC output rate. Receiver gain
does not scale noise. Integrated acquisition returns `RoundStatistics` with joint
state counts and channel totals. Decimated acquisition returns `RoundSignals`,
reordering point-major samples into QICK loop order. The adapter owns packing,
rounding, deferred raw generation and averaging.

The model assumes weak dispersive readout, independent weak-coupling Markov
secular baths and exponential echo-characteristic dephasing. Truncations and
numerical tolerances remain user choices. Conditional means include multilevel
relaxation, not random jump times. Variance includes initial-state mixture,
receiver noise and the integrated rounding term.

`private_replay=absolute_directory` opts into checked private input archives.
`replay_round(round_path)` recomputes one round without drivers, earlier rounds,
current YAML or numeric records. Backend, engine numeric version, RNG version and
dependency versions must match. The stream policy is `PCG64/SeedSequence-v2`.
Old exact-record manifests fail explicitly. Archives contain model truth and
must not appear in ordinary Run/native/RPC metadata. Normal provenance remains
the opaque simulation identifier.

`plan_point` and `build_database` remain transitional author tools for the old
continuous solver and offline library. Normal acquisition does not call them.
Their tail-frozen readout and exact-record semantics do not describe the template
runtime. They are removed with that older numeric stack.
