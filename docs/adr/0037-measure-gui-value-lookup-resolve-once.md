# ADR-0037 — measure-gui read-only value lookup and resolve-once references

**Status:** Accepted

**Context:** measure-gui adapter defaults and agent/UI write paths sometimes need a
small amount of global session information whose source varies by workflow:
cached device values, active project labels, loaded predictor metadata, or
operator-selected flux device state. Stable needs still use explicit typed
helpers and ports. A string-key lookup is only an escape hatch for rare,
multi-source defaults and for writing the current value of a source into
`MetaDict`.

**Related:** ContextService is the single md/ml write authority in [[0006]];
service role and port discipline follow [[0004]]/[[0005]]; session-core sharing
is defined in [[0020]]; cfg editor sessions and `EvalValue` wire handling are
defined in [[0008]]; role defaults and CfgBuilder live under [[0009]]/[[0012]].

## Decision

measure-gui provides a read-only value lookup in the session layer.

- `ValueLookup` is the read interface: callers list metadata or resolve a known
  key as a typed scalar.
- `ValueRegistry` is the registration interface: only the session composition
  root and source owners mutate it.
- `ExpContext` may carry the read-only lookup facade so adapter default
  generation can use it without receiving app services or mutable registries.
- Provider registration is owner-scoped. Owners can atomically replace all their
  providers or unregister as a group when a predictor reloads, a device changes,
  or a context is replaced.
- Providers are synchronous, fast, side-effect-free, and read-only. Device
  providers read cached `SessionState.devices[*].info`; they never poll hardware.
- Lookup errors distinguish missing keys, known-but-unavailable values, provider
  failures, duplicate registration, and type mismatches.
- Built-in owners project current session state only: context/project labels,
  loaded predictor metadata, and cached named-device values such as
  `device.<name>.name` and `device.<name>.value`. The device layer does not
  publish `active_flux`; flux meaning belongs to the experiment cfg/adapter that
  chooses a device.

The lookup does not replace explicit typed dependencies. Existing stable helpers
such as `proper_res_freq_range()`, `proper_qub_freq_range()`, `proper_relax()`,
and direct `ctx.md`/`ctx.ml` reads remain the preferred path.

## Resolve-once input

Source references use a sibling concept to `EvalValue`, not an extension of
`EvalValue`.

- `EvalValue` remains a live md expression stored in cfg value trees and resolved
  by cfg lowering.
- `ValueRef` means "read this registered value now and materialize the result".
  It is never persisted as a lazy reference.
- Agent wire uses explicit tagged objects, for example
  `{"__kind":"value_ref","key":"device.flux.value","type":"float"}` where
  `flux` is a concrete registered device name.
- GUI text conveniences use the shared `value_source_input.py` controller:
  typing `@{prefix` opens a segment-scoped completion popup (`@{` lists top-level
  segments, `dev` completes `device`, `device.` lists immediate children).
  Pressing Tab/Backtab accepts the default candidate; when every current candidate
  is a namespace segment rather than a registered leaf key, the controller appends
  `.` and immediately refreshes the popup for the next segment. A complete
  registered key closes with `}`. A trailing space after `@{full.key} ` resolves
  once and replaces the token with the current value formatted as text. Plain
  strings in the wire contract are not globally interpreted.
- Adapter default generation may use `CfgBuilder.value_ref(...)`; role-default
  seeds may use `Source(...)`. Both resolve through `ExpContext.values` during
  default construction and store ordinary direct values in the cfg tree.

`ContextService` remains the md/ml write authority. Any path that writes a
`ValueRef`-style token to `MetaDict` resolves it in the UI/input layer before
calling `coerce_md_value` and `set_md_attr`, so the service sees and stores an
ordinary scalar.

## Expression boundary

The first implementation supports token replacement, not a general composite
source-expression parser. After a GUI token is resolved, any surrounding text is
handled by the existing target input: cfg scalar eval mode keeps the resulting
expression text, while Inspect md coercion accepts only scalar literals.

If composite resolve-once expressions become necessary, the existing
`gui.session.expression` AST numeric evaluator is deepened into a restricted
numeric resolver with md-name, source-token, and small function resolvers. The
design does not use Python `eval` or SymPy as the default engine: GUI scalar
inputs need deterministic numeric results, not symbolic algebra or a broad
function surface.

## Consequences

- Default generation can use a small escape hatch without importing app services
  into experiment adapters.
- Agent and GUI entry points share one lookup but expose different surfaces:
  agents get explicit tagged objects; GUI gets segment-scoped completion and
  resolve-on-space token replacement.
- Owner-level unregister avoids stale provider closures after device/predictor
  lifecycle changes.
- The design adds a string-key registry, so usage must remain narrow and
  documented. Broad migration from explicit helpers to lookup keys is a
  regression in readability and testability.
