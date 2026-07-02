# Architecture Decision Records

每篇 ADR 以現在式描述目前生效的跨模組設計。索引只放定位用摘要；細節、替代方案與演化脈絡留在 ADR 本文。

程式碼註解與記憶檔以 `ADR-NNNN` 引用本目錄；ADR 之間以 `[[NNNN]]` 互鏈。

## Concurrency / Lifecycle

- [0001 — Permit / Lease typed guard](0001-permit-lease-typed-guard.md)：靜態前置憑證與動態硬體互斥分離。
- [0002 — Version table + async handle + off-main handler](0002-version-table-async-handle-off-main.md)：GUI resource version guard、operation handle、off-main wait 三層分工。
- [0003 — ShutdownCoordinator and registry cancel](0003-shutdown-coordinator-and-registry-cancel.md)：統一 cancel/poll/await 詞彙與 Qt-free shutdown loop。
- [0019 — Operation facets and execution strategy](0019-operation-facets-and-execution-strategy.md)：Operation 由 Exclusion、Handle、Progress、Cancel facet 組合。
- [0025 — Cross-thread interaction channel](0025-cross-thread-interaction-channel.md)：operation/user prompt 使用單一有序 channel 傳遞 settle、message、stop。
- [0026 — OperationRunner + scope ports](0026-operation-abstraction-runner-scope-ports.md)：OperationRunner 擁有通用生命週期；各 operation 只提供 policy 與窄 write port。

## GUI Service Architecture

- [0004 — Service dependency three questions](0004-service-dependency-three-questions.md)：用 Query / Command / Reaction 判斷 service 依賴方向。
- [0005 — Service roles](0005-service-roles-ddd-hexagonal.md)：Driving adapter、app service、aggregate root、repository、driven adapter 的角色邊界。
- [0006 — Single ml/md write authority](0006-single-ml-md-write-authority.md)：`ContextService` 是 ModuleLibrary / MetaDict 內容寫入權威。
- [0007 — Device state lives in State](0007-device-state-to-state-ssot.md)：Device live state 由 State 擁有，DeviceService 保持 driver/worker 邊界。
- [0020 — Shared session core](0020-session-core-shared-layer.md)：measure 與 autofluxdep 共用 context、SoC、device、dialog、operation/session primitive。
- [0021 — Event ownership domain modules](0021-event-ownership-domain-modules.md)：事件 enum 與 payload 由 domain module 擁有，app 只組裝 bus 與 serializer。
- [0037 — Value lookup + resolve-once refs](0037-measure-gui-value-lookup-resolve-once.md)：session value source 提供少量 default / md-write escape hatch；`ValueRef` 立即 materialize。

## Cfg / Value Model

- [0008 — CfgEditor session](0008-cfg-editor-session.md)：GUI widget 與 agent 共用 service-owned LiveModel draft。
- [0009 — Spec/Value fluent + LiteralSpec lock](0009-spec-value-fluent-and-literal-lock.md)：Spec tree 靜態、Value tree 可變；locked literal 只在 spec 宣告。
- [0010 — Complete value tree + None for empty](0010-value-tree-complete-none-for-empty.md)：Value tree 永遠完整；optional empty 統一用 `None`。
- [0011 — CfgSchema validate boundary](0011-cfgschema-validate-boundary.md)：成品邊界做靜態結構驗證。
- [0012 — CfgBuilder value assembly](0012-cfgbuilder-value-layer-fluent-assembly.md)：adapter default value 透過 fluent builder 與 role table 組裝。
- [0036 — Adapter capability contract](0036-adapter-capability-contract-validated-at-import.md)：adapter 顯式宣告 capabilities，import-time validation 抓宣告與 hook 不一致。

## Remote / Transport

- [0013 — RemoteControlAdapter as second view](0013-remote-adapter-as-second-view.md)：remote socket 是 MainWindow 平級 driving adapter。
- [0014 — Shared GUI transport layer](0014-gui-shared-transport-layer.md)：三個 GUI app 共用 NDJSON RPC endpoint 與 MCP bridge primitive。

## Persistence

- [0015 — PersistenceCaretaker](0015-persistence-caretaker-memento-single-file.md)：GUI app-state 用單一 memento file，由 caretaker 管理讀寫時機。
- [0027 — Experiment data persistence](0027-experiment-data-persistence-native-labber-axes-list.md)：Experiment data file 使用 Labber axes-list、typed axes spec、grouped dataset roles。
- [0039 — QubitParams owns params.json](0039-qubit-params-json-owner.md)：`meta_tool.QubitParams` 是 result-scope `params.json` 的 typed 讀寫權威。

## Experiment Runtime

- [0038 — Executor ResultTree](0038-executor-result-tree.md)：executor workflow 使用 ResultTree、per-measurement update event、template-method lifecycle 與 MeasurementBundle contract。

## Analysis / Simulation / Waveform

- [0028 — Fluxdep analysis kernel](0028-fluxdep-analysis-kernel.md)：flux-dependence analysis kernel 位於 GUI / notebook adapter 之外。
- [0029 — Fluxonium prediction engine](0029-fluxonium-prediction-engine.md)：Fluxonium prediction policy 位於 `simulate.fluxonium`。
- [0030 — Arbitrary waveform optional recipe](0030-arbitrary-waveform-asset-optional-recipe.md)：arbitrary waveform asset 是 qubit-scoped `.npz`，可內嵌 formula recipe。
- [0031 — Formula recipe segments](0031-formula-recipe-complex-segments.md)：formula recipe 使用 ordered segments 與 complex expression。
- [0032 — Reference time axis](0032-arbitrary-waveform-reference-time-axis.md)：arbitrary waveform playback 使用 asset 自帶 time axis。
- [0033 — Delete/rename without reference scan](0033-arbitrary-waveform-delete-no-reference-scan.md)：asset mutation 不掃描或遷移 ModuleLibrary references。
- [0034 — ArbWaveformDatabase repository](0034-arb-waveform-database-shared-asset-repository.md)：shared asset operation 收斂到 `meta_tool.ArbWaveformDatabase`。
- [0035 — MCP failures use tool errors](0035-arb-waveform-mcp-failures-use-tool-errors.md)：arb waveform MCP validation/missing/collision 走 failed tool call 與 stable reason。

## Plotting

- [0016 — notebook liveplot auto close](0016-notebook-liveplot-auto-close-default.md)：notebook liveplot 預設 auto-close，不依賴 ipympl 私有協議。
- [0017 — Worker-thread plotting](0017-worker-thread-plotting.md)：worker 直接畫圖時 marshal；只通知時走 queued signal。

## Autofluxdep / Agents

- [0018 — Autofluxdep resolver builder](0018-autofluxdep-orchestrator-requirement-resolver-builder-currying.md)：autofluxdep node/service 共用 requires/provides/produce 介面。
- [0022 — Worktree coordination](0022-agent-coordination-worktree.md)：多 agent / 長線 orchestration 使用 `.agent_state/` worktree protocol，主 checkout merge 由 merge queue 序列化。
- [0023 — Cooperative interrupt feedback](0023-cooperative-interrupt-feedback-wakeup.md)：由 [[0025]] 取代；保留為被取代設計的定位點。
- [0024 — Agent launch UI retirement](0024-embedded-agent-session-architecture.md)：measure-gui 不內建 Agent launch UI；agent 啟動由外部 CLI/MCP workflow 負責。
