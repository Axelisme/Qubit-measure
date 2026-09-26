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
- [0058 — Registry-owned VISA session disconnect](0058-registry-owned-visa-session-disconnect.md)：device 局部契約見 [device README](../../lib/zcu_tools/device/README.md)；跨 owner teardown 歸 Operation，故本篇尚未退役。

## GUI Service Architecture

- [0004 — Service dependency three questions](0004-service-dependency-three-questions.md)：用 Query / Command / Reaction 判斷 service 依賴方向。
- [0005 — Service roles](0005-service-roles-ddd-hexagonal.md)：Driving adapter、app service、aggregate root、repository、driven adapter 的角色邊界。
- [0006 — Single ml/md write authority](0006-single-ml-md-write-authority.md)：`ContextService` 是 ModuleLibrary / MetaDict 內容寫入權威。
- [0007 — Device state lives in State](0007-device-state-to-state-ssot.md)：Device live state 由 State 擁有，DeviceService 保持 driver/worker 邊界。
- [0020 — Shared session core](0020-session-core-shared-layer.md)：measure 與 autofluxdep 共用 context、SoC、device、dialog、operation/session primitive。
- [0021 — Event ownership domain modules](0021-event-ownership-domain-modules.md)：事件 enum 與 payload 由 domain module 擁有，app 只組裝 bus 與 serializer。
- [0048 — Domain event facts and View reactions](0048-domain-event-facts-and-view-reactions.md)：producer發布closed domain fact；pane-owned State先commit完整resource並於commit後回收retired draft，coordinator擁有lazy-snapshot reaction matrix與figure restore政策。
- [0037 — Value lookup + resolve-once refs](0037-measure-gui-value-lookup-resolve-once.md)：session value source 提供少量 default / md-write escape hatch；`ValueRef` 立即 materialize。
- [0064 — GUI process startup 與 app composition](0064-process-startup.md)：launcher、runtime、app startup coordination、plotting 與 remote 的責任分界。
- [0053 — Owner scheduler 與 gate presence](0053-owner-scheduler-and-gate-presence.md)：core 以 `OwnerScheduler` port 取代 Qt main-thread 隱含假設，service completion 全走 EventBus；hardware gate lease 附 origin_kind/note/since 供多前端 presence。

## Cfg / Value Model

- [0008 — CfgEditor session](0008-cfg-editor-session.md)：GUI widget與agent共用service-owned `CfgDraft`；Analysis/Post各自持有不洩漏`editor_id`的opaque writeback draft。
- [0009 — Spec/Value fluent + LiteralSpec lock](0009-spec-value-fluent-and-literal-lock.md)：Spec tree 靜態、Value tree 可變；locked literal 只在 spec 宣告。
- [0010 — Complete value tree + None for empty](0010-value-tree-complete-none-for-empty.md)：Value tree 永遠完整；optional empty 統一用 `None`。
- [0011 — CfgSchema validate boundary](0011-cfgschema-validate-boundary.md)：成品邊界做靜態結構驗證。
- [0012 — Context-free measure cfg definition](0012-cfgbuilder-value-layer-fluent-assembly.md)：adapter以單一definition宣告static shape、domain verbs與deferred typed defaults。
- [0036 — Adapter capability contract](0036-adapter-capability-contract-validated-at-import.md)：adapter顯式宣告capabilities，import-time validation抓宣告與hook不一致，Load所有driving paths共用`load_data` gate。
- [0045 — Shared GUI cfg core ownership](0045-shared-gui-cfg-core-ownership.md)：`gui.cfg` 擁有Qt-free core，`gui.widgets.cfg`擁有instance-registry Qt renderer，measure adapter與autoflux cfg barrel只暴露app-owned API；lowering ports見 [[0046]]。
- [0046 — Shared cfg lowering ports](0046-shared-cfg-lowering-ports.md)：finished-cfg algorithm由shared core擁有，app以expression/reference/range三個窄port提供runtime policy。
- [0050 — Canonical cfg binding paths](0050-canonical-cfg-binding-paths.md)：binding擁有唯一typed path grammar與batch net diff；remote只投影target。
- [0051 — Canonical program cfg shape catalog](0051-canonical-program-cfg-shape-catalog.md)：`gui.measure_cfg`擁有closed program shape vocabulary與raw materialization policy；`gui.cfg`只提供domain-free spec walk。

## Remote / Transport

- [0013 — RemoteControlAdapter as second view](0013-remote-adapter-as-second-view.md)：remote socket 是 MainWindow 平級 driving adapter。
- [0014 — Shared GUI transport layer](0014-gui-shared-transport-layer.md)：三個 GUI app 共用 NDJSON RPC endpoint 與 MCP bridge primitive。
- [0047 — Typed expected-error taxonomy](0047-typed-expected-error-taxonomy.md)：caller-correctable failure 由 producer 以 closed category 顯式 opt in，transport 只投影。
- [0049 — Subscriber-aware lazy push](0049-subscriber-aware-lazy-push.md)：endpoint以two-phase recipient transaction在matching subscriber存在時才materialize/encode一次，並維持unsubscribe/disconnect線性化。
- [0052 — Event meta 與多前端 attribution](0052-event-meta-and-frontend-attribution.md)：bus 為事件蓋章 `EventMeta(seq, origin)`，origin 由 dispatch 邊界宣告、operation 記錄顯式攜帶；coalescing 屬 subscriber-side；wire 封套 additive 加 seq/origin。
- [0059 — measure MCP RPC channel](0059-measure-mcp-rpc-channel.md)：低頻 wire method 經 live GUI 提供的 `rpc.catalog` 與通用 `rpc_*` 呼叫；exposure 與 guard policy 隨 method 宣告於 `RemoteMethodEntry`。
- [0060 — Agent interface as second view](0060-measure-agent-interface-shared-gui-view.md)：量測 agent 以 40 個特化 tool 操作與 GUI 共用的狀態；一個判斷點一個 tool，寫入類 tool 使 GUI 跟隨到對應子 tab。
- [0061 — Measure interactive plugin session](0061-measure-interactive-plugin-session.md)：plugin 局部契約見 [main app README](../../lib/zcu_tools/gui/app/main/README.md)；跨 owner session ownership 歸 GUI，故本篇尚未退役。

## Persistence

- [0063 — Persistence 保存權威與資料契約](0063-persistence-ownership.md)：區分 memento、experiment result、run artifact、參數、樣品座標與波形資產的 owner、完整性、失敗與引用邊界；局部格式見各 owner 文件。

## Experiment runtime／Autofluxdep workflow

- [0062 — 實驗執行與 workflow 編排](0062-experiment-workflow.md)：runtime、executor、GUI Node、run snapshot、feedback 與 RunSession 的責任分界。


## Analysis / Simulation / Waveform

局部分析／預測契約見 [fluxdep](../../lib/zcu_tools/analysis/fluxdep/README.md)、[fitting](../../lib/zcu_tools/analysis/fitting/README.md) 與 [prediction](../../lib/zcu_tools/simulate/fluxonium/README.md)；waveform 資產契約見 [repository 參考](../../lib/zcu_tools/resources/waveform_assets.md)。

## Plotting

Notebook liveplot 關閉與 backend 契約見 [liveplot README](../../lib/zcu_tools/plotting/liveplot/README.md)。

- [0017 — Worker-thread plotting](0017-worker-thread-plotting.md)：worker 直接畫圖時 marshal；只通知時走 queued signal。

## Agents

協作流程歸外部 dev-flow／collab skills，repo 的 live resource 限制見 [CLAUDE.md](../../CLAUDE.md)。

- [0018 — Autofluxdep resolver builder](0018-autofluxdep-orchestrator-requirement-resolver-builder-currying.md)：保留 Builder／Node 與 requires/provides/produce 原介面；§3 的 predictor 校正與載入敘述已被取代（現行 overlay 見 0062，按需載入的目標見 draft）。
- [0023 — Cooperative interrupt feedback](0023-cooperative-interrupt-feedback-wakeup.md)：由 [[0025]] 取代；保留為被取代設計的定位點。

## Draft

- [Autofluxdep 逐項宣告依賴與 predictor 載入](draft/autofluxdep-explicit-dependencies.md)：已確認待實作，不代表現行契約。
- [外部 agent launch 責任](draft/external-agent-launch-ownership.md)：已核准方向待 Remote／Transport ADR 核實轉正。
- [Agent operation feedback](draft/agent-operation-feedback.md)：已核准方向待 Operation ADR 核實轉正。

## Retired

以下舊篇保留原號與正文；現行 Persistence 決策見 0063，workflow 決策見 0062，process startup 決策見 0064。

- [0044 — GUI process runtime](retired/0044-gui-process-runtime.md)：跨模組啟動邊界由 0064 接替，局部契約見 gui README。
- [0024 — Agent launch UI retirement](retired/0024-embedded-agent-session-architecture.md)：有效的 launch／feedback 邊界暫見 Remote／Operation draft。
- [0015 — GUI memento caretaker](retired/0015-persistence-caretaker-memento-single-file.md)：app 與 shared caretaker 的責任由 0063 接替。
- [0027 — Experiment data persistence](retired/0027-experiment-data-persistence-native-labber-axes-list.md)：資料責任由 0063 接替，細節在 datafile 與 experiment owner 文件。
- [0032 — Waveform reference time axis](retired/0032-arbitrary-waveform-reference-time-axis.md)：時間權威由 0063 接替。
- [0033 — Waveform reference lifecycle](retired/0033-arbitrary-waveform-delete-no-reference-scan.md)：不級聯更新由 0063 接替。
- [0038 — Executor ResultTree](retired/0038-executor-result-tree.md)：共用 ResultTree 與執行骨架見 `lib/zcu_tools/experiment/v2/runtime/README.md`；autofluxdep executor 與 flux tracker 契約見 `lib/zcu_tools/experiment/v2/README.md`，workflow collection 邊界見 0062。
- [0039 — QubitParams JSON owner](retired/0039-qubit-params-json-owner.md)：typed handoff 由 0063 接替。
- [0040 — Autofluxdep run artifact](retired/0040-autofluxdep-run-result-artifact.md)：保存邊界由 0063 接替，workflow lifecycle 見 0062。
- [0057 — SampleTable v2](retired/0057-flat-sampletable-v2-coordinate-contract.md)：座標保存契約由 0063 接替，schema 細節由 sample_table owner 維護。
- [0041 — Autofluxdep feedback framework](retired/0041-autofluxdep-feedback-framework.md)：feedback 跨模組邊界接入 0062，slot 細節移至 app README。
- [0042 — Autofluxdep feedback confidence reversion](retired/0042-autofluxdep-feedback-confidence-reversion.md)：freshness 邊界接入 0062，公式移至 app README。
- [0043 — Autofluxdep runtime cfg override plan](retired/0043-autofluxdep-runtime-cfg-override-plan.md)：cfg 邊界接入 0062，snapshot／decoration 細節移至 app README。

以下舊篇已分流至模組 owner 文件；退役正文僅供追溯。

- [0054 — Resonance multiplicative amplitude background](retired/0054-resonance-multiplicative-amplitude-background.md)：real log-amplitude slope 乘完整 resonator response，`edelay` 維持唯一 global phase slope。
- [0055 — Route-scoped resonator electrical delay](retired/0055-route-scoped-resonator-electrical-delay.md)：boundary search 可 bounded adaptive expansion；成功 delay 與 generator/readout route 以單一 compound calibration 持久化為後續 branch seed。
- [0056 — Resonance phase curvature and rational initializer](retired/0056-resonance-phase-curvature-rational-initializer.md)：optional resonance-centered quadratic phase background 與 internal degree-1 rational initializer 不接管 route-scoped delay branch。
- [0028 — Fluxdep analysis kernel](retired/0028-fluxdep-analysis-kernel.md)：flux-dependence analysis kernel 位於 GUI / notebook adapter 之外。
- [0029 — Fluxonium prediction engine](retired/0029-fluxonium-prediction-engine.md)：Fluxonium prediction policy 位於 `simulate.fluxonium`。
- [0030 — Arbitrary waveform optional recipe](retired/0030-arbitrary-waveform-asset-optional-recipe.md)：arbitrary waveform asset 是 qubit-scoped `.npz`，可內嵌 formula recipe。
- [0031 — Formula recipe segments](retired/0031-formula-recipe-complex-segments.md)：formula recipe 使用 ordered segments 與 complex expression。
- [0034 — ArbWaveformDatabase repository](retired/0034-arb-waveform-database-shared-asset-repository.md)：資產操作現由 `resources.ArbWaveformDatabase` 擁有。
- [0035 — MCP failures use tool errors](retired/0035-arb-waveform-mcp-failures-use-tool-errors.md)：arb waveform error 由 main app 投影到 remote 契約。
- [0016 — notebook liveplot auto close](retired/0016-notebook-liveplot-auto-close-default.md)：notebook liveplot 預設 auto-close，不依賴 ipympl 私有協議。
- [0022 — Worktree coordination](retired/0022-agent-coordination-worktree.md)：舊協作 protocol；通用流程改由外部 skill 管理，本地 live resource 限制見 CLAUDE.md。
