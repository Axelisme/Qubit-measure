# autofluxdep-gui

A control-type GUI that runs an automated flux-dependence workflow: a sweep over
flux × an ordered list of measurement Nodes, each Node declaring its
dependencies and the information it produces. Replaces the notebook-style
`experiment.v2.autofluxdep` cfg-maker lambdas + `FluxDepEnv.info` handoff chains
with an explicit, declarative model.

The mechanisms the lower-layer module gets right are kept (in-place
accumulation, sweep-lived plotters fed one flux point at a time, result merging).
The GUI still keeps orchestration separate from acquisition: a Node is NOT a
runner object and does not inherit an execution protocol. The kind of provider is
a **Builder** (one subclass per experiment) that, per flux point, produces a
short-lived **Node** with a narrow `produce` interface; the orchestrator drives
those uniformly.

The **orchestrator** sees only declarations plus `produce`: `requires`,
`requires_modules`, `provides`, `provides_modules`, and the Node entrypoint. It
is a pure **requirement resolver** — NOT an ordering /
topological resolver. Execution order is the user's explicit list order; the
orchestrator just runs it. Per flux point, for each provider in order: it
projects `requires` against the current info/module state into a snapshot
(latest-available, with skip/fallback), calls `produce(snapshot) -> Patch`, and
merges the Patch by the declared output contract. Execution resources (soc,
Result, tools, round_hook, plot-side notifications, etc.) are opaque values in
`RunEnv`: the orchestrator constructs and passes that environment through, but
domain decisions about drawing, acquire, fit, calibration, and feedback remain
inside the Builder/Node. This keeps the resolver surface narrow and lets it call
`produce` uniformly on every provider — zero `isinstance`, no distinguishing a
Node from a Service.

**Run path (real acquire).** The app composes the shared session services
(`gui/session`: connection / context / device / startup) and uses the shared
setup / device / predictor dialogs; `State(AutoFluxDepState)` inherits
`SessionState` and the run reads the active `exp_context`. Each measurement
Node's `produce` builds the real run cfg from that context (`Builder.make_cfg`
→ `ml.make_cfg` lowering), writes this flux point's value into
`cfg.dev[flux_device]` by device **name** (`set_flux_by_name` — the GUI picker
stores a device name, e.g. the auto-provisioned `fake_flux`; the lower layer's
`flux_dev` label is a different dimension), pushes it with `setup_devices`, then
runs the experiment program's `.acquire` (TwoToneProgram / ModularProgramV2 /
…) with a running-average `round_hook` + `stop_checkers` (cooperative cancel +
SNR early-stop), and fits — `qubit_freq` defaults to fixed-bias residual
feedback and only hard-bias mode feeds `predictor.calibrate`; `ro_optimize`
takes an argmax, `mist` reads the variance, both
without a fit. There is **no synthetic fallback**: `make_cfg` Fast Fails
(`RuntimeError`) when the context is unconfigured, and the orchestrator turns a
`produce` exception into a terminal `RunFailedPayload` (the run worker QThread is
never aborted). Offline, the acquire runs against the **flux-aware MockSoc**
(`connect_mock` provisions `fake_flux`); since the SimEngine reads the operating
flux live, the acquired signal varies with the swept flux. The shared
real-acquire helpers (set-flux / stop-checkers / fit-quality gate / axis parse)
live in `nodes/acquire.py`.

## Language

**Builder**:
The kind of provider — one subclass per experiment (qubit_freq, ro_optimize, t1,
predictor, ...), **stateless**. It holds the type-level declarations (provides /
requires), the sweep-lived factories called once at Run start
(`make_init_result`, `make_plotter`), and a per-flux-point factory `build_node`.
The Builder is how the **environment is curried into the Node**: the execution
layer calls `build_node(this point's snapshot / soc / Result / round_hook /
Plotter / tools)`, and the Builder closes that environment into the returned
Node, whose `produce` then takes only the dependency input. Using a Builder
(rather than the experiment class implementing the Provider protocol directly)
is precisely what keeps `produce` narrow: the alternative — an experiment class
with `produce(snapshot, soc, result, round_hook, ...)` — would fatten the
interface and leak execution environment into what the orchestrator sees, or
push that state onto a stateful experiment object. Currying via Builder avoids
both.
_Avoid_: NodeSpec, NodeType (the kind is a Builder/factory, not a spec or a
behaviour-holding base).

**Node**:
What a **Builder produces for one flux point** — short-lived (built each point,
discarded after), holding that point's execution state (its detunes, cfg) with
the sweep environment curried in. Its only orchestrator-facing surface is
`produce(snapshot) -> Patch` (plus its declared provides / requires, copied from
the Builder). A measurement Node's `produce` runs acquire (round_hook fills the
sweep Result's row + notifies) then fit; the sweep-lived Result / Plotter it
fills were closed in by `build_node`.
_Avoid_: "the Node holds state across flux points" — it is one point; the Result
/ Plotter / tools hold the cross-point state.

**Service**:
A provider whose Builder produces a Node by **pure computation, not hardware**
(the predictor's Builder produces a Node computing `predict_freq` / `cur_m`).
Same `produce` interface to the orchestrator; the difference is only what its
Builder curries in (no soc / Result / round_hook / Plotter — pure compute draws
and stores nothing) and that it is not in the user's Node list. So a Service
participates in requirement resolution through the same three interfaces, never
masquerading as a measurement Node nor stubbing out unused behaviours. A Service
leaking into the orchestrator (a pre_point seeding `predict_freq`) or pre-loaded
into Tools regardless of need was a real coupling bug: a Service is loaded only
because some Node requires what it provides.

**Builder behaviors / operations**:
The methods a Builder defines (make_init_result / make_plotter / build_node /
make_cfg) — domain operations the execution layer *calls*, NOT callbacks. The
cfg-derivation step (the active context + this point's snapshot → a runnable
experiment cfg) IS a per-experiment Builder method, `make_cfg(env, snapshot)`,
but it **runs inside the Node's `produce`** — where the resolved snapshot (the
predicted freq, the latest-available modules) is in hand (the Builder owns the
recipe; the Node invokes it when it has the dependency input). `make_cfg` lowers
the context's ml/md + the drive params into the experiment's typed cfg via
`ml.make_cfg` (the same lowering the lower-layer cfg_maker lambdas did). Reserve
"callback" / "hook" for genuine event callbacks the worker invokes (round_hook /
the row-updated notify). The **Builder is stateless**: sweep-lived execution
state lives in the Result (detunes/cfg) / Plotter (plot lines) / tools
(learners), and a single point's state lives in the short-lived Node that
`build_node` produces. A stateful Builder (`self.detunes`, `self.freq_line`)
would conflate the kind with one execution.

**Sweep**:
One full Run: the orchestrator iterates flux points, and at each flux point runs
every Node in list order. The unit of "the whole run", as opposed to a single
flux point or a single Node execution.

**Run Result Artifact**:
The canonical persisted evidence for one autofluxdep Sweep. It is run-scoped,
not node-scoped: it ties the workflow definition, per-Node committed result
rows, dependency Patch events, skips, failures, and terminal status to one run
identity. Fluxdep / dispersive handoff files and reports are exports from this
artifact, not separate sources of truth.
_Avoid_: export file, report bundle, memento.

**Committed Node Row**:
The durable statement that one Node completed its measurement attempt for one
flux point and its Result row is safe to read back. A committed row may still
have an empty Patch when fitting or providing failed; that means "measured but
did not provide", not "unmeasured".
_Avoid_: successful fit, completed flux point.

**Flux Point Commit**:
The durable statement that the workflow finished processing a flux point at the
provider-loop level. It is distinct from individual Node row commits: a flux
point can have some committed node rows and still fail or stop before the full
point commits.
_Avoid_: row commit, node completion.

**Run Journal Event**:
A machine-readable audit event in the Run Result Artifact, such as node row
written, node skipped, node failed, flux point committed, or run finalized.
Patch and skip/failure state live here because they describe workflow dependency
state, not a Node's measurement data channel.
_Avoid_: log line, Labber channel, debug message.

**Tools**:
The container of **general, sweep-lived, stateful services** curried into Nodes
by their Builder (predictor, feedback runtime, ...). Services hold the
cross-flux-point state that a short-lived Node cannot. Per-node private services
built by a `make_services` hook were rejected as over-design — a capability used
by only one Builder today is still a general Tools service.
_Avoid_: per-node service, make_services.

The **predictor service** has two faces. *Query* (general): `predict_freq(flux)`
and `predict_matrix_element(flux)` — the predictor predicts both, used by the
predictor Node to produce base `predict_freq` / `cur_m`. *Calibration* (a service
method triggered by a Node, NOT by the orchestrator): `calibrate(flux,
measured_freq)` — qubit_freq uses it only when its `bias_update_mode` is `hard`,
handing a trusted measured freq to the service to adjust the physical/base
prediction when the backend supports it. Fixed-bias mode leaves the raw predictor
unchanged. The predictor does **not** hide residual IDW correction; qubit_freq
owns composition of `base predict_freq + correction`, and the correction
estimator is a generic feedback slot in `Tools.feedback`. The orchestrator never
calibrates and never updates feedback slots itself.

**Feedback capability**:
A run-lived, placement-scoped map of generic scalar estimators/controllers,
built from Builder-declared slots and the placed node's Generation overrides.
It is exposed to a Node as `RunEnv.feedback`. The generic layer provides only
mechanics: `idw` / `last_good` estimators and a `log_step` controller. Estimates
and proposals are `FeedbackSample(value, confidence, age_queries)` objects; the
generic layer decays confidence as a function of query age since the last
trusted observation/proposal, but it never invents a domain fallback. It does
not know what the scalar means, does not emit Patch keys, and does not apply
fit gates, clamps, bounds, stop/fail policy, or fallback defaults. A disabled
declared slot returns `None`; an undeclared slot lookup fast-fails.
_Avoid_: feedback patch, node-private controller state, orchestrator feedback.

**Result**:
A Node's domain output, distinct from its **Patch**. It is **sweep-lived and
flux-aware**: a Node knows the workflow sweeps flux, so its Result carries the
flux axis directly, filled in place one flux row at a time. This is unlike a
generic list-and-merge runner result: the Node has direct flux workflow context,
so there is no per-point result list to merge.
_Avoid_: signals (too generic); "per-point result" (the Result spans the sweep).

The **flux axis is always the first dimension**, but the trailing dimensions and
the drawing differ by Node type — which is exactly why `make_init_result` and
`make_plotter` are Builder methods, not generic. qubit_freq's Result is
`(n_flux, n_detune)` and its Plotter *accumulates* (a flux×detune colormap grown
over the sweep); ro_optimize's Result is `(n_flux, n_freq, n_gain)` and its
Plotter *overwrites* (only the current flux row's freq×gain map is shown, peak
marked). "Plotter lives the whole Sweep" does NOT dictate accumulate vs
overwrite — the Plotter gets `(result, idx)` and decides whether to draw all
flux rows or only row idx; that is domain knowledge. (ro_optimize also produces
a module — the tuned readout — into its Patch alongside the best-freq/gain
scalars; the contract is unchanged, a Patch already carries modules.)

**Patch** vs **Result**:
Two projections of one fit. The **Result** is *this Node's own complete output*
— the raw 2D signals, per-point freq axes, fit curves, cfg — used by the Plotter
and for saving. The **Patch** is *what other Nodes consume* — the provides
scalars for the dependency system (e.g. `qubit_freq=5001.5`). Result = for
saving and drawing (self); Patch = for downstream Nodes (others). A measurement
Node's `produce` fits once per flux point, then fills both — they cannot
disagree because they come from the same computation.

Patch modules follow the same public-contract rule: produced modules must be
concrete ModuleLibrary-lowerable dictionaries. Required module dependencies do
not get placeholder defaults; if neither an upstream Node nor ML alias provides
the module, the resolver skips the Node with a missing-module reason.

A Patch may be **partial**: when the fit is poor, `produce` simply omits that
provides key (does NOT write nan). A downstream Node then reads the *latest
available* value — it falls back to the previous flux point automatically, with
zero checking at the call site. Writing nan instead would force every downstream
reader to `isnan`-check and fall back itself; omitting the key keeps that out of
the call sites. So `validate_patch` requires the Patch's keys to be a *subset* of
`provides` (not exactly equal). The Result still gets its row filled (raw
present, fit fields nan). (Fit-quality judgement and cross-point state like
"steps since last success" are domain logic inside `produce`; the fit-quality
gate (`is_good_fit`) discards a noisy / dead flux point — its `produce` returns a
partial Patch, omitting the key.)

The Builder pre-allocates the empty (nan-filled) Result at Run start via
`make_init_result(params, n_flux)` — its flux extent is known (`n_flux`) and its
detune extent comes purely from the `detune_sweep` param (length is param-only;
the per-point detune *values* shift with `predict_freq` and are filled per row).
A Node's `produce` then fills the Result's flux-idx row **in place** (the Result
was curried in by `build_node`) and returns only the **Patch**.

Detune is a hard sweep: one acquire returns the whole 1-D detune trace, and the
**round_hook** is called each round with the running-averaged whole trace. So
`produce` *overwrites* the Result's flux-idx row with the current whole trace
each round (the same row grows clearer round by round, while earlier flux rows
are already settled), then — only after acquire+fit complete — fills that row's
fit fields (`fit_freq[idx]`, fit curve). The fit fields stay nan during acquire,
so the Plotter must tolerate a mid-acquire row that has raw signals but no fit.

On **stop**, the Result keeps its nans — it is not truncated, matching the
Schedule/executor convention of preserving preallocated nan-filled result data.
Rows after the stop stay nan (an honest "not measured"); a row interrupted
mid-acquire keeps its partial round-averaged raw (fit still nan). No special
handling — the nan pre-allocation + the Plotter's nan tolerance already cover it.

**Plotter**:
A Node-type-defined, stateful object that draws that Node's figure. Its
lifetime is the whole **Sweep** (built once at Run start, fed the Node's
flux-aware **Result** as it fills, redrawn each flux point), NOT a single Node
execution. It holds the drawing state (line / colormap objects) but never owns
the Qt widget. Built by `Builder.make_plotter(figure)`, where `figure` is a
plain matplotlib `Figure` (embedded with a bare `FigureCanvasQTAgg` in the
Node's run tab — NOT via gui/plotting's FigureContainer/backend, since that
marshal layer is for worker-drawn figures and this worker never draws; see
ADR-0018). At **Run start** the UI builds, for every workflow Node, its figure +
Plotter + Result together (all three are sweep-lived), so auto-follow can switch
to any Node's plot at any time.
_Avoid_: live plot (that's the rendering substrate, `zcu_tools.liveplot`);
"per-execution plotter" (the lifetime is the sweep, not one execution).

The Plotter is **never marshalled** (ADR-0017 does NOT apply): the worker NEVER
touches matplotlib. The worker fills the Result's flux-idx row in place (numpy)
and emits a plain notification signal (the flux index, no figure); a main-thread
slot then calls `plotter.update(result, idx)`. All drawing — Plotter, figure,
Result reads — stays on the main thread. (ADR-0017 marshals worker threads that
draw directly; here the worker only notifies, so the simpler Qt
queued-signal notification suffices. The shared numpy Result is safe because the
worker only writes row idx and the main thread reads it only after the
queued signal, which gives happens-before.)

The **whole sweep runs on one worker thread** — `build_node` + the Node's
`produce` (derive-cfg → acquire → fit → fill) included. Unlike measure-gui, where
`run` (acquire + liveplot raw) and `analyze` (fit) are split across two workers,
autofluxdep's fit is part of the sweep loop: each flux point must fit immediately
to feed `predict_freq` / the predictor and the downstream Nodes' dependencies, so
acquire and fit cannot be separated. The main thread only: builds figure +
Plotter + Result at Run start, redraws on each row-updated notification, and
reacts to run-lifecycle events.

**Ownership.** The per-Node **Results** live in `State.run_results` (the
accumulated domain data — read by the Plotter, by saving, by the Info dialog —
so it goes in State, even though it is not serialisable). The **Plotters** are
UI-owned (only the UI redraws). The main thread builds the empty Result
container at Run start (State stores the finished container); the worker then
fills its rows in place. Filling pre-allocated numpy content is **not a State
semantic write** — it does not add/remove keys or swap objects, does not bump a
version or emit, exactly like Schedule/executor result buffers filled in place by
their worker. So it does not violate the "State writes only on the main thread"
invariant, which governs structural/semantic changes (add_node / set_flux), not
row-fills of a container the main thread already placed.
