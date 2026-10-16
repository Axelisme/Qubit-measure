# `gui.app.measure.remote`

**Last updated:** 2026-10-16，logical recipe RPC

This package is the GUI-process side of measure-gui remote control. It exposes a
local NDJSON RPC surface over the live `Controller`, injects the application's control identity into shared remote dispatch,
serializes selected events, and pushes diagnostics.

The agent-facing MCP bridge lives in `zcu_tools.mcp.measure`. This package does
not declare MCP tools and does not own stdio transport.

## Owners

- [`service.py`](service.py)：`RemoteControlAdapter`, a measure-gui subclass of shared
  `RemoteControlServiceBase`; injects control authority, and owns
  writeback key aliases and diagnostics.
- [`dispatch.py`](dispatch.py)：runtime method registry projection; keeps the public
  `METHOD_REGISTRY` import path stable.
- [`handlers/`](handlers/)：grouped wire method handlers bound to controller, control facets,
  or render-view calls.
- [`method_specs.py`](method_specs.py)：Qt-free projection for wire method schema and timeouts.
- [`method_entries/`](method_entries/)：single registration source for method name, handler ref,
  `MethodSpec`, agent exposure, readonly and operation policy and `ParamSpec` shorthands. The
  `rpc.catalog` projection uses those same entries; handler refs are resolved
  only by dispatch.
- [`events.py`](events.py)：domain payload type to wire event serializer mapping.
- [`form_observation.py`](form_observation.py)：tab Form input/values/errors projection and
  the shared path-array edit decoder.
- [`form_description.py`](form_description.py)：single-layer Form description and field observation。
- [`form_errors.py`](form_errors.py)：將 measure 的 CfgPreconditionError 轉譯為 remote 錯誤；
  來源 provenance 與 precondition 型別由 services/form_contracts.py 擁有。
- [`wire_version.py`](wire_version.py)：measure-gui wire contract version and GUI code revision.

Shared transport primitives live in [`zcu_tools.gui.remote`](../../../remote/README.md): NDJSON framing,
typed request/reply/error envelope, socket endpoint, owner-scheduler dispatch, and
router scaffolding. Typed envelopes live in [`gui.remote.wire`](../../../remote/wire.py).

## Wire contract

- One connection has at most one in-flight RPC.
- Request and response roots are JSON objects.

## Recipe 欄位

`tab.read_recipe_field`、`tab.edit_recipe_fields` 與 `tab.select_recipe_module` 明確使用 logical paths。`form_description.py` 擁有 RecipeFieldObservation、RecipeEditResult 與 RecipeSelectionResult。所有 description、errors 與 source paths 都是 logical。通用 describe_cfg/edit_cfg/get_cfg/reset_cfg 保持 canonical 契約，read_module 保留明確 materialization。

Read 使用一次 owner turn 的目前 input 與固定 capture。Value_status 區分 available、missing 與 unavailable，input_present/input 只觀察 authored draft。Selector 描述目標本身，source 識別最近 captured ref；readonly 表示穿過 ref 祖先，不代表 declaration lock。讀取不改 input、base、cfg version 或通知，也不授權寫入。

Edit 保留 ordered list 與原始 index，owner 一次檢查路徑與 admission。結構、source、ref ancestor 或 guard 拒絕不提交前綴。一般 invalid draft 可回 changed=true 加 logical preview errors。Select 回 changed 與同 turn 的 fresh field。Use_library 必須帶非空 actual key，其他 action 必須省略 key；explicit null 不等於省略。

Handler 將 LogicalEditError 投影為 invalid_params 與原 reason，保留 path、edit_index、selector_path、available_keys。FormPathError 保留 form_path，LogicalSourceError 回 precondition_failed/library_source，並帶 consuming path 與來源身分。Cfg/control 失敗沿既有投影。ParamSpec primitive/required/enum 失敗維持原生 reason；handler carrier 與 action/key 語義失敗才回 form_errors。Unexpected provider/declaration failures 不重分類。

## Simulation startup

Simulation environments are installed before remote readiness with
`--simulate <environment.py>`. GUI runtime environment switching and
`simulation.initialize` are retired. RPC cannot replace the environment;
changing it requires a restart. See the
[simulation contract](../../../../program/v2/simulation/README.md#build-and-own-an-environment)
for the startup entry point and ownership.

## Authority and dispatch

After `wire.version` and optional `auth`, send `control.connect` with
`{label: "agent", mode: "owner"}` (both values default as shown). Before this
command a socket is an observer: explicit readonly methods and `rpc.catalog`
are available, business writes return `not_holder`, and acquire/release return
`readonly_connection`. Initialization fixes label and mode for that socket.
An identical repeat is inert; a different label or mode returns `invalid_params`.
Reconnect to change identity. Label is a trusted LAN identity, not a credential.

Unmarked methods require holder at owner-loop execution, before domain effects and existing domain guards. `tab.interact` remains unmarked, including reads.
A `not_holder` error includes `{holder, locked, online}`. `operation.await` is
readonly; a new holder can cancel accepted work. Subscription methods only
change connection-local observation subscriptions and remain available to
observers. Measure and Fluxdep share this authority; Dispersive stays readonly.
Cfg commands use current inputs without refs. See ADR-0070 for authority and GUI gates.

Normal handlers run on the State owner thread through the injected `OwnerScheduler`.
The Qt composition root supplies `QtOwnerScheduler`; headless tests use
`ManualOwnerScheduler`. Handler exceptions become typed error envelopes.

Only bounded wait handlers run off-main:

- `operation.await`

Off-main handlers only consume thread-safe channels. They do not read or mutate
owner-thread state, version tables, controller objects or Qt widgets.

## Completion and provenance

Reads return current owner projections; they do not establish a per-connection
mutation baseline. Holder admission replaces seen maps and resource-version
comparison. Domain guards, result provenance, busy checks and lifetime still
apply. RPCs are forwarded once, without hidden refresh, retry or rollback.
Timeout or response failure may follow an accepted mutation.

Diagnostics are separate from EventBus. The controller pushes diagnostics to the
remote adapter sink, which broadcasts diagnostic payloads to clients regardless
of subscription. Measure MCP does not subscribe, queue or piggyback events.
Agent-visible async completion comes from operation request/reply, not pushes.

`tab.save_data` and `tab.save_artifacts` complete synchronously on the GUI owner
thread. They return the current complete Save report with the original Run ID,
even when every output fails, and every output path/status/error. Output statuses
and errors determine save success, not the presence of `run_id`. Readonly tab
snapshots retain confirmed identity: null until a persistent output succeeds.
They do not return a Save handle or require
`operation.await` followed by `tab.snapshot`. Optional Run/Analyze source IDs
still bind the result being saved; they are not Save IDs.

## Contract references

- [Shared GUI](../../../README.md)：frame/queue limits and response-delivery failures。
- [`method_entries/`](method_entries/)：per-method parameters, replies, timeouts and agent policy。
- [`handlers/`](handlers/)：wire payload and error projection。
- [`wire_version.py`](wire_version.py)：wire handshake and version bump rules。
- [ADR-0070](../../../../../../docs/adr/0070-control-identity.md)：control identity and GUI gates。
- [ADR-0068](../../../../../../docs/adr/0068-remote-transport.md)：transport、expected errors、events 與 GUI／MCP 分工。
- [ADR-0065](../../../../../../docs/adr/0065-cfg-editing.md)／[ADR-0008](../../../../../../docs/adr/0008-cfg-editor-session.md)：Form 與 writeback。
- [ADR-0066](../../../../../../docs/adr/0066-operation-lifecycle.md)／[ADR-0063](../../../../../../docs/adr/0063-persistence-ownership.md)：operation、shutdown、provenance 與 persistence。
- [Measure framework](../README.md)／[MCP bridge](../../../../mcp/measure/README.md)：application owners 與 agent-facing policy。
