# `zcu_tools.gui.remote` — shared remote adapter

**Last updated:** 2026-10-24

**讀者：** framework-dev。共用 NDJSON RPC 傳輸、排程與連線契約。

This package owns NDJSON framing, typed envelopes, authentication, socket queues,
owner-scheduler dispatch and diagnostic delivery. Apps supply methods
and application authority; transport does not own experiment policy.
`RemoteAppSpec` groups each app's handshake metadata, method registry and reply
continuations. Socket options, owner scheduler and optional control
authority remain separate constructor inputs.

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

`NdjsonRpcEndpoint.reply_ok(on_complete=...)` binds a single reply's terminal
send callback at enqueue. `ReplyDelivery.DELIVERED` means every byte of the success frame
was sent to the OS socket, not that the peer acknowledged or consumed it.
Encoding/enqueue/send failure and close/stop complete with `ReplyDelivery.FAILED`. Callbacks run
once outside transport locks on the completing thread and must not block.
`RemoteAppSpec.reply_continuations` maps accepted methods to owner-thread actions.
Shared dispatch posts these actions after terminal send, including failure;
handler rejection does not run an action. No new timeout or wire message is added.

Response failure or timeout can follow a committed command. Transport neither
rolls back domain effects nor retries requests. Controller diagnostics retain
their independent broadcast and queue/backpressure behavior.

`ParamSpec.minimum` / `maximum` project inclusive numeric JSON-schema limits.
These declarations do not add generic range admission: domain owners validate
ranges and provide the same constants to schemas, including dependent limits
in descriptions. Undeclared-field behavior stays unchanged.

`ParamSpec.nullable=True` admits explicit null as `None`, independently of
whether the key is required. Schemas retain required keys and add `"null"` to
the declared type and enum when present. JSON parameters keep their untyped
schema. The default `False` preserves existing null handling: required values
reject null, while optional values use their declared default. Missing optional
keys still use that default, even when nullable is True.

`ParamSpec.omit_when_missing=True` 讓 optional、default=None 的欄位在 absent 時不進 handler params，explicit null 仍以 None 傳入。Handler 可用 membership 判斷有沒有提供該欄位。Required 或帶非 None default 的宣告在建構時拒絕此設定。預設 False 保持原有 absent/null/default 行為，schema requiredness 仍由 required 決定。

Measure startup selection belongs to the launch composition, not transport.
MCP forwards a trusted script only when launching a new GUI; attach never reruns
startup. Shared remote does not load simulation environments or restore domain
sessions. Script failures and script-origin device notification policy remain
with the measure app, while operation facts and outcomes retain their origin.

Session discovery stores one `<app>.json` per running GUI under
`utils.cache.user_cache_root()/sessions`. GUI writers and MCP readers must use
the same user and cache-root environment on the same host. Set
`ZCU_TOOLS_CACHE_DIR` to a non-empty absolute root path to isolate both discovery
and GUI persistence. Empty or relative overrides raise ValueError.
Platform defaults and the tempdir fallback are documented in the
[utils README](../../utils/README.md#user-cache-root). Discovery does not read the
old `~/.cache/zcu-tools/sessions` location.

Wire details live in the [measure remote README](../app/measure/remote/README.md).
App composition and error boundaries are in
[ADR-0068](../../../../docs/adr/0068-remote-transport.md).

Local modal dialogs refuse owner acquisition with `local_busy` and "Local 使用中".
Qt app adapters query the active modal on the owner turn; shared control remains
Qt-free. Readonly and existing Remote holder reconnects keep their contracts.
