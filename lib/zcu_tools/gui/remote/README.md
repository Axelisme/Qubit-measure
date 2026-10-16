# `zcu_tools.gui.remote` — shared remote adapter

**Last updated:** 2026-10-16，optional parameter omission

This package owns NDJSON framing, typed envelopes, authentication, socket queues,
owner-scheduler dispatch and lazy event delivery. Apps supply methods, serializers
and application authority; transport does not own experiment policy.

Measure and Fluxdep inject the same `ControlIdentity` used by their Qt frontend.
Business methods require holder by default; explicit readonly declarations allow
observers. Control management and catalog remain callable with their own admission
rules. Dispatch checks current holder on the owner loop before domain effects.
Fixed socket identity, trusted labels, lock/reclaim and structured refusals are
specified in [ADR-0070](../../../../docs/adr/0070-control-identity.md).

Reads return current projections, not mutation baselines. MCP does not keep a
second authority or version table. Authentication, exposure, domain validation,
busy checks and source-bound operations remain independent. Readonly apps do not
inject writable control authority.

RPC remains the app's basic control interface. Fluxdep MCP composes several RPCs
into a decision tool and summarizes the app's full projections. The app owns
prerequisites, drafts, publication, fit freshness and rendering; shared remote does
not infer them. Readonly render methods return evidence without opening a Session.
The same owner renderer supplies the GUI and MCP evidence.

Response failure or timeout can follow a committed command. Transport neither
rolls back domain effects nor retries requests. Event subscriptions keep their
connection-local lifecycle and lazy serialization/backpressure behavior.

`ParamSpec.minimum` / `maximum` project inclusive numeric JSON-schema limits.
These declarations do not add generic range admission: domain owners validate
ranges and provide the same constants to schemas, including dependent limits
in descriptions. Undeclared-field behavior stays unchanged.

`ParamSpec.omit_when_missing=True` 讓 optional、default=None 的欄位在 absent 時不進 handler params，explicit null 仍以 None 傳入。Handler 可用 membership 判斷有沒有提供該欄位。Required 或帶非 None default 的宣告在建構時拒絕此設定。預設 False 保持原有 absent/null/default 行為，schema requiredness 仍由 required 決定。

Wire details live in the [measure remote README](../app/measure/remote/README.md).
App composition and error boundaries are in
[ADR-0068](../../../../docs/adr/0068-remote-transport.md).

Local modal dialogs refuse owner acquisition with `local_busy` and "Local 使用中".
Qt app adapters query the active modal on the owner turn; shared control remains
Qt-free. Readonly and existing Remote holder reconnects keep their contracts.
