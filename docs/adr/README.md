# Architecture Decision Records

每篇 ADR 以現在式描述目前生效的跨模組設計。索引只放定位用摘要；細節、替代方案與演化脈絡留在 ADR 本文。

程式碼註解與記憶檔以 `ADR-NNNN` 引用本目錄；ADR 之間以 `[[NNNN]]` 互鏈。

ADR 宣告 package 之間的依賴方向時，同時在 `.importlinter` 建立對應 contract，並在本文寫出 contract ID。沒有 contract 的依賴規則只能靠判斷維持，視為 ADR 未完成。

## Concurrency / Lifecycle

- [0066 — Operation 執行、取消與關閉](0066-operation-lifecycle.md)：guard／lease／handle、runner、互動、shutdown 與 device disconnect 的跨 owner 責任；尚未落實的核准目標見 [Operation draft](draft/operation-lifecycle-boundaries.md)。
- [0001 — Permit / Lease typed guard](0001-permit-lease-typed-guard.md)：measure typed Permit 的局部 guard 契約仍有效；跨 owner 分界由 0066 承接。
- [0002 — Version table + off-main handler（handle 模型沿革）](0002-version-table-async-handle-off-main.md)：資源版本 guard 與 RPC off-main handler 的局部／Remote 契約仍有效；operation handle 分界由 0066 承接。
- [0053 — Owner scheduler 與 gate presence](0053-owner-scheduler-and-gate-presence.md)：scheduler port、State owner guard、completion facts 與 gate presence 的局部契約仍有效；operation owner-loop 規則由 0066 承接。

## GUI Service Architecture

- [0067 — GUI 應用核心與前端邊界](0067-gui-application.md)：shared session、owner、capability、domain fact、前端反應與 explicit plotting 責任。
- [0064 — GUI process startup 與 app composition](0064-process-startup.md)：launcher、runtime、app startup coordination、rendering 初始化與 remote 的責任分界。

## Cfg / Value Model

- [0065 — Cfg 編輯模型與使用邊界](0065-cfg-editing.md)：共用機制、measure tab resource 的原子 publication／固定 Run acceptance、Custom reference 繼承及獨立 draft 的現況分界；library conversion、selected Apply 與其他 app 的剩餘目標見 [Cfg draft](draft/cfg-editing-boundaries.md)，measure slice 的設計與驗收邊界見 [resource contract](draft/cfg-resource-contract.md)。
- [0008 — Measure CfgEditor session](0008-cfg-editor-session.md)：measure headless session 與 opaque writeback 的局部契約仍有效；其中 tab auto-commit／雙樹不是共用目標。
- [0009 — Spec/Value fluent + LiteralSpec lock](0009-spec-value-fluent-and-literal-lock.md)：Spec／Value 和角色預設的局部契約，概要見 [cfg owner](../../lib/zcu_tools/gui/cfg/README.md)。
- [0010 — Complete value tree](0010-value-tree-complete-none-for-empty.md)：完整 Value tree 與停用表示的局部契約。
- [0011 — Finished-cfg validation](0011-cfgschema-validate-boundary.md)：成品驗證的局部條件；舊篇所述 app-local lowering 位置已更新。
- [0012 — Context-free measure definition](0012-cfgbuilder-value-layer-fluent-assembly.md)：實驗 adapter 的 definition／seed 及 assembler 契約。
- [0037 — Session value lookup](0037-measure-gui-value-lookup-resolve-once.md)：lookup 與 resolve-once 輸入的 measure 局部契約。
- [0045 — Shared GUI cfg core](0045-shared-gui-cfg-core-ownership.md)：renderer registry／import surface 的局部細節仍有效；舊 program owner 路徑已過時。
- [0046 — Shared cfg lowering ports](0046-shared-cfg-lowering-ports.md)：generic lowering 操作與 ports；live-key 說明不得推定為核准的 refresh／override 目標。
- [0050 — Canonical cfg binding paths](0050-canonical-cfg-binding-paths.md)：target grammar／diff 的局部契約；成功前綴是現況而非核准的 atomic batch 目標。
- [0051 — Program cfg shape catalog](0051-canonical-program-cfg-shape-catalog.md)：program shape／raw materialization 局部契約；目前 owner 已移至 `experiment.cfg_editing`。

## Remote / Transport

- [0068 — 遠端前端、傳輸與 agent 介面](0068-remote-transport.md)：四個 app 的共用傳輸和 app policy 分界、事件與錯誤投影、measure 固定工具及 live catalog、穩定名稱寫回，以及 analysis-only JSON 投影。尚未實作的 GUI 收合見 [draft](draft/remote-event-coalescing.md)。

## Persistence

- [0063 — Persistence 保存權威與資料契約](0063-persistence-ownership.md)：區分 memento、experiment result、run artifact、參數、樣品座標與波形資產的 owner、完整性、失敗與引用邊界；參數容器使用工作單位與顯式定義注入，局部格式見各 owner 文件。

## Experiment runtime／Autofluxdep workflow

- [0062 — 實驗執行與 workflow 編排](0062-experiment-workflow.md)：runtime、executor、GUI Node、run snapshot、feedback 與 RunSession 的責任分界。


## Analysis / Simulation / Waveform

局部分析／預測契約見 [fluxdep](../../lib/zcu_tools/analysis/fluxdep/README.md)、[fitting](../../lib/zcu_tools/analysis/fitting/README.md) 與 [prediction](../../lib/zcu_tools/simulate/fluxonium/README.md)；waveform 資產契約見 [repository 參考](../../lib/zcu_tools/resources/waveform_assets.md)。

## Plotting

Notebook liveplot 關閉與 backend 契約見 [liveplot README](../../lib/zcu_tools/plotting/liveplot/README.md)；GUI 的 worker／Qt 繪圖責任見 0067。

## Agents

協作流程歸外部 dev-flow／collab skills，repo 的 live resource 限制見 [CLAUDE.md](../../CLAUDE.md)。

- [0018 — Autofluxdep resolver builder](0018-autofluxdep-orchestrator-requirement-resolver-builder-currying.md)：保留 Builder／Node 與 requires/provides/produce 原介面；§3 的 predictor 校正與載入敘述已被取代（現行 overlay 見 0062，按需載入的目標見 draft）。

## Draft

- [模組定位與依賴邊界](draft/module-boundaries.md)：各 package 的定位、五層、穩定工具與有狀態模組的分類、`cfg_model` 共享核心與債務清單。分層與無循環規則由 `.importlinter` C8–C13 檢查，C15 保護使用者 core 的 GUI independence，C16 禁止 framework 反向 import 使用者套件；其餘債務尚未清除。
- [實驗核心、前端包裝與具名圖形產物](draft/experiment-interface-redesign.md)：全實驗與 caller 已在 task 分支遷移，舊自訂 pyplot routing backend 已退場。修正候選已完成軟體驗證及雙軸審查；正式接受以 task 的逐列證據與裁決紀錄為準，landing 另需使用者授權。草案不代表持久分支已採用。包含 records、具名原生圖、RunContext／CfgEnv、instance DeviceManager 與模擬環境 coordinator 的分工。

- [Autofluxdep 逐項宣告依賴與 predictor 載入](draft/autofluxdep-explicit-dependencies.md)：已確認待實作，不代表現行契約。
- [GUI 事件收合](draft/remote-event-coalescing.md)：GUI 訂閱端依 payload type 與 tab 的 per-tick 收合尚未落實；cfg 表單 snapshot 收合已實作。
- [Cfg 編輯接縫與使用邊界](draft/cfg-editing-boundaries.md)：measure tab 的 resource、refresh／override、revision 與 atomic batch 已落實；library-entry editing port、selected Apply 與其他 app 仍待收斂。
- [Cfg 資源公開契約與同步發布](draft/cfg-resource-contract.md)：measure tab 的 resource-bound editing、可寫節點、同步 publication、expression capture 與 explicit revision 設計及驗收記錄；其他 app 未因這條接線完成而全面遷移。
- [GUI adapter capability 入口檢查](draft/gui-adapter-capability-guards.md)：analysis／post-analysis application 入口的完整拒絕尚待實作。
- [Operation 關閉與 disconnect](draft/operation-lifecycle-boundaries.md)：shutdown 期限、無 handle 背景工作與 GUI device/factory owner 的未落實目標。

## Retired

以下舊篇保留原號與正文；現行 GUI 決策見 0067，Operation 見 0066，Remote／Transport 見 0068，Persistence 見 0063，workflow 見 0062，process startup 見 0064。Cfg 來源篇尚有有效局部契約，未整篇退役。

- [0013 — Remote adapter second view](retired/0013-remote-adapter-as-second-view.md)、[0014 — Shared GUI transport](retired/0014-gui-shared-transport-layer.md)、[0047 — Expected errors](retired/0047-typed-expected-error-taxonomy.md)、[0049 — Lazy push](retired/0049-subscriber-aware-lazy-push.md)、[0052 — Event attribution](retired/0052-event-meta-and-frontend-attribution.md)、[0059 — MCP RPC catalog](retired/0059-measure-mcp-rpc-channel.md)、[0060 — Measure agent interface](retired/0060-measure-agent-interface-shared-gui-view.md)：跨模組邊界由 0068 接替；0052 未落實的 GUI 收合見 draft。
- [0004 — Service dependency three questions](retired/0004-service-dependency-three-questions.md)：依賴語意與方向由 0067 接替。
- [0005 — Service roles](retired/0005-service-roles-ddd-hexagonal.md)：前端與 owner 角色由 0067 接替。
- [0006 — Single ml/md write authority](retired/0006-single-ml-md-write-authority.md)：寫入權威由 0067 接替；cfg lowering 見 0065 與 measure README。
- [0007 — Device state lives in State](retired/0007-device-state-to-state-ssot.md)：device 與 persistence 分工由 0067、0063 接替。
- [0017 — Worker-thread plotting](retired/0017-worker-thread-plotting.md)：前端繪圖邊界由 0067 接替，機制見 gui plotting README。
- [0020 — Shared session core](retired/0020-session-core-shared-layer.md)：共用／app-local 邊界由 0067 接替，局部契約見 session README。
- [0021 — Event ownership domain modules](retired/0021-event-ownership-domain-modules.md)：domain 事件與前端投影由 0067 接替。
- [0036 — Adapter capability contract](retired/0036-adapter-capability-contract-validated-at-import.md)：跨 owner 規則由 0067 接替，精確 hook 見 experiment adapter README。
- [0048 — Domain event facts and View reactions](retired/0048-domain-event-facts-and-view-reactions.md)：事件與反應由 0067 接替，pane matrix 見 measure README。
- [0061 — Measure interactive plugin session](retired/0061-measure-interactive-plugin-session.md)：session ownership 由 0067 接替，互動細節見 measure README。
- [0003 — Shutdown coordinator](retired/0003-shutdown-coordinator-and-registry-cancel.md)：取消與 shutdown 跨 owner 分界由 0066 承接；自動逾時關窗敘述不再是核准的共用預設。
- [0019 — Operation facets](retired/0019-operation-facets-and-execution-strategy.md)：facets 與 execution 分界由 0066 承接，局部策略見 session／app README。
- [0023 — Cooperative interrupt feedback](retired/0023-cooperative-interrupt-feedback-wakeup.md)：舊 FeedbackInbox 由 0066 的有序 channel 取代。
- [0025 — Cross-thread interaction channel](retired/0025-cross-thread-interaction-channel.md)：互動分界由 0066 承接；局部事件契約見 session／app README。
- [0026 — OperationRunner and ports](retired/0026-operation-abstraction-runner-scope-ports.md)：runner 分界由 0066 承接；局部 port 與 scope 見 session／app README。
- [0058 — Registry-owned disconnect](retired/0058-registry-owned-visa-session-disconnect.md)：跨 owner teardown 由 0066 承接，registry 局部契約見 device README。
- [0044 — GUI process runtime](retired/0044-gui-process-runtime.md)：跨模組啟動邊界由 0064 接替，局部契約見 gui README。
- [0024 — Agent launch UI retirement](retired/0024-embedded-agent-session-architecture.md)：外部 agent 啟動邊界由 0068 接替；operation feedback 見 0066。
- [0015 — GUI memento caretaker](retired/0015-persistence-caretaker-memento-single-file.md)：app 與 shared caretaker 的責任由 0063 接替。
- [0027 — Experiment data persistence](retired/0027-experiment-data-persistence-native-labber-axes-list.md)：資料責任由 0063 接替，細節在 datafile 與 experiment owner 文件。
- [0032 — Waveform reference time axis](retired/0032-arbitrary-waveform-reference-time-axis.md)：時間權威由 0063 接替。
- [0033 — Waveform reference lifecycle](retired/0033-arbitrary-waveform-delete-no-reference-scan.md)：不級聯更新由 0063 接替。
- [0038 — Executor ResultTree](retired/0038-executor-result-tree.md)：共用 ResultTree 與執行骨架見 `lib/zcu_tools/experiment/v2/runtime/README.md`；autofluxdep executor 與 flux tracker 契約見 `zcu_lab/v2/experiment-authoring.md`，workflow collection 邊界見 0062。
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
