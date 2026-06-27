# Architecture Decision Records

每篇 ADR 以**現在式**描述目前生效的設計；被取代/推翻的設計只保留簡述 + 推翻原因（折進對應 ADR 的「演化」段，不另立檔）。檔名編號依主題分組排序，非建立時間序。

> 程式碼註解與記憶檔以 `ADR-NNNN` 引用本目錄；ADR 之間以 `[[NNNN]]` 互鏈。本目錄已入 git 追蹤（同 CLAUDE.md「文件追蹤」），會進 diff 與 commit。

## I. 並發、守衛與生命週期

- [0001 — Permit / Lease typed guard](0001-permit-lease-typed-guard.md)：型別強制的 Permit（呼叫前可靜態證明的前置）與 OperationLease（動態 hardware 互斥）分離。
- [0002 — 版本表 + async handle + off-main handler](0002-version-table-async-handle-off-main.md)：並發感知＝資源版本表 guard（非追 origin）+ RPC-as-proxy operation handle + off-main blocking handler + 三層分工。（併入舊 origin-tracking 與 change-buffer 的演化）
- [0003 — 統一 async cancel + ShutdownCoordinator](0003-shutdown-coordinator-and-registry-cancel.md)：Registry 持 stop_event（cancel/poll/await 三動詞齊備）+ Qt-free 輪詢關閉。（§一「綁死」被 [[0019]] 取代）
- [0019 — Operation = token + opt-in facets + 可插 execution strategy](0019-operation-facets-and-execution-strategy.md)：Exclusion/Handle/Progress/Cancel 四 facet 任意組合 + strategy（OffMain-thread/pool、Main-thread-user-paced、Blocking）多型；拆 Handle 出 gate、抽 BackgroundService、「async in main thread」。取代 [[0003]] §一綁死。
- [0025 — 跨線程互動 channel](0025-cross-thread-interaction-channel.md)：GUI↔agent 跨線程互動改用**單一 per-interaction 有序事件 channel**（typed `Settled`/`Message`/`Stop(reason)`，consumer 依到達序折疊）取代「completion Event + feedback inbox + stop_event」三 channel 的時序敏感 combine；producer 不阻塞、consumer 限時阻塞 → race-free（全序）+ deadlock-free（單向有界等待）by construction；「Send & Stop」視為單一 `Stop(reason)` 意圖；含 run/analyze/device/connect/notify_user 的適用性分析（save 因同步無 handle 不適用）。取代 [[0023]]，關聯 [[0019]]/[[0017]]/[[0026]]。
- [0026 — operation abstraction：OperationRunner + scope-as-adapter + State write ports](0026-operation-abstraction-runner-scope-ports.md)：抽 kind-agnostic `OperationRunner`（唯一生命週期機制，組合非繼承）+ 各 op 交 `OperationSpec` policy（領域邏輯 interpret/on_terminal 留各 op）；`BackgroundService` 退化純執行器、scope-wiring 進 adapter（`ActiveTask` 不再外洩 bg）；State→窄 write port；gate/progress 維持 port 兄弟不併入；`ConnectionService` 拆 SoC+Predictor、DeviceService 保留+`DeviceRegistryPort`、save 留抽象外 + agent handle 外露 / 泛型 op poll-wait（START 顯式回 handle、wait/poll 吃 handle 直打 op-agnostic wire、cancel 維持 op-specific、product fold 移入 START/getter）。關聯 [[0019]]/[[0025]]/[[0004]]/[[0005]]/[[0017]]/[[0007]]。

## II. Service 架構

- [0004 — Service 互依的三問規則](0004-service-dependency-three-questions.md)：Query/Command/Reaction 三問取代分層表治循環；含「狀態放 State vs service 私有」兩軸。
- [0005 — Service 角色規範（DDD + Hexagonal）](0005-service-roles-ddd-hexagonal.md)：Driving Adapter / App Service / Aggregate Root / Repository / Driven Adapter 五角色 + 重構結果。
- [0006 — ml/md 內容寫入的唯一權威 = ContextService](0006-single-ml-md-write-authority.md)：經窄 write port，lowering+register 收進 ContextService；run-time experiment cfg materialization 從 `ModuleLibrary` store 拆到 stateless `assemble_experiment_cfg`，`make_cfg` 只是薄 wrapper。
- [0007 — Device 狀態下放 State（SSOT）](0007-device-state-to-state-ssot.md)：DeviceService 退化 driver/worker；persistence 為 State 投影。
- [0021 — Event 所有權：domain module 擁有 enum + payload](0021-event-ownership-domain-modules.md)：domain module 擁有 enum+payload 定義；app 在 bus/EVENT_SERIALIZERS 層組裝；bus 維持 payload-type-key；port 集中各層 ports.py（掛 [[0004]]/[[0005]]）。
- [0037 — measure-gui read-only value lookup + resolve-once refs](0037-measure-gui-value-lookup-resolve-once.md)：session-layer `ValueLookup` / owner-scoped `ValueRegistry` 作為少數 default / md-write escape hatch；`ValueRef` 立即 materialize，不擴充 `EvalValue`；adapter defaults 經 `CfgBuilder.value_ref` / role `Source` 窄用。

## III. CfgEditor session

- [0008 — CfgEditor session](0008-cfg-editor-session.md)：service-owned headless LiveModel + 可插拔 widget viewer；writeback 草稿的 agent 編輯入口統一在 writeback 面（`editor_id` 內部化、新增 `CfgEditorPort.set_field`），ml-entry 編輯仍走外露 editor_id 的 `editor.*`。（併入 headless-only / delegated 的演化）

## IV. Spec / Value 設定模型

- [0009 — Spec/Value fluent + LiteralSpec 鎖定 + 角色 default 表（ROLE_TABLE）](0009-spec-value-fluent-and-literal-lock.md)
- [0010 — Value 樹永遠完整 + None 統一表「空」](0010-value-tree-complete-none-for-empty.md)：（併入 DisabledRefValue marker 的演化）
- [0011 — CfgSchema.validate 成品邊界靜態檢查](0011-cfgschema-validate-boundary.md)
- [0012 — CfgBuilder value 層組裝 builder](0012-cfgbuilder-value-layer-fluent-assembly.md)
- [0036 — Adapter capability 契約（顯式 `AdapterCapabilities` + import-time 驗證）](0036-adapter-capability-contract-validated-at-import.md)：每 adapter 顯式宣告 `AdapterCapabilities(analysis/requires_soc/post_analysis)`，`__init_subclass__` 在 import 時 getattr-identity（MRO-aware、不 hardcode base）Fast-Fail 宣告↔hook 不一致；analyze-params override 只在 params 無法全 default 建構時必須（否則 base 回 `params_cls()`）。取代 design_v2 的 DEC-1/3/4 機制探索，關聯 [[0009]]/[[0012]]。

## V. Remote / 傳輸

- [0013 — RemoteControlAdapter 作為第二個 View](0013-remote-adapter-as-second-view.md)：與 MainWindow 平級的 driving adapter + 診斷 off-bus fan-out + ViewProtocol 三拆。
- [0014 — GUI 三 app 共用純傳輸層](0014-gui-shared-transport-layer.md)：NdjsonRpcEndpoint + McpBridge 共用；policy 留各 app。

## VI. 持久化

- [0015 — PersistenceCaretaker（Memento + Caretaker）](0015-persistence-caretaker-memento-single-file.md)：單檔 app-state、關閉才寫。
- [0027 — 實驗資料持久化：labber_io 原生 axes-list + per-experiment axes-spec + grouped experiment dataset](0027-experiment-data-persistence-native-labber-axes-list.md)：**（accepted/部分已落地）** 刪 datasaver dict 殼（含洩漏軸序的 load-flip）；public API 收斂到 `zcu_tools.utils.datasaver` package facade，caller 透過 re-exported `save_labber_data` / `load_labber_data` 使用 inner-first axes-list（N 維、load 為 save 恒等逆、零 transpose）；每實驗一份 typed axes-spec 驅動共用 save/load helper。Grouped Experiment Dataset 延伸採單一 Experiment Data File + 多 Labber log group 表達多個 Dataset Role；legacy artifact 只透過 migration script 轉換，不保留 runtime compatibility path。與 [[0015]] 劃清（app-state vs 實驗資料）。

## VI-A. 分析 kernel

- [0028 — Flux-Dependence Analysis kernel lives outside notebook and GUI adapters](0028-fluxdep-analysis-kernel.md)：Flux-Dependence Analysis 的互動選點 / filtering / line selection / one-tone peak detection domain implementation 住 notebook-neutral `zcu_tools.analysis.fluxdep`；notebook 與 Qt GUI 只是 adapters。第一批只含 stateless 計算函式 + interaction state machine，不含 database search / plotting / export。
- [0029 — Fluxonium Prediction engine owns simulation policy outside GUI adapters](0029-fluxonium-prediction-engine.md)：Fluxonium Prediction 的 value-to-flux affine、typed resolution、dispersive fast/scqubits fallback、fallback provenance 與 axis-bound cache identity 住 `zcu_tools.simulate.fluxonium`；GUI/session/notebook 只是 adapters。`FluxoniumPredictor` 保留穩定 facade 並可逐步委派到 engine。
- [0030 — Arbitrary waveform assets allow optional formula recipes](0030-arbitrary-waveform-asset-optional-recipe.md)：Arbitrary Waveform Asset 是 qubit-scoped single-file `.npz` asset；Data Key 必須符合 `^[A-Za-z][A-Za-z0-9_]*$`；archive 只允許 `idata`/`qdata`/`time`/optional `recipe_json`，extra key 以 `unknown_npz_key` fail，invalid embedded `recipe_json` 也 fail；`idata`/`qdata`/`time` arrays 必備且同 shape 1D finite、`2 <= len(time) <= MAX_ARB_WAVEFORM_SAMPLES = 1_000_001`、`time[0] == 0`、time strictly increasing、imported raw time 不要求等間距、`Abs = sqrt(I^2 + Q^2)` 在 `[0, 1]`；超過上限以 `sample_count_too_large` fast-fail；Formula Recipe 以 optional `recipe_json` UTF-8 JSON scalar 內嵌；derived metadata 一律即時計算不持久化，normalization scale 不顯示。External file import 只接受含 `idata`/`qdata`/`time` 的 `.npz`，不支援 `.npy`、CSV/clipboard/text table。
- [0031 — Formula recipes use complex expressions and ordered segments](0031-formula-recipe-complex-segments.md)：Formula Recipe 使用單一 SymPy-compatible expression；real output 對 I、complex output 拆 I/Q。Recipe JSON 只含 required `segments` 與 required `normalize`，`segments` 必須至少一段、每段 `duration > 0` 且 `formula.strip()` 後不可空，validation/parse 使用 stripped expression 但存檔保留原始 formula 字串，`normalize` 只允許 `"none"` / `"peak"`；不含 `schema_version`、`kind`、`time_unit`、`notes`；所有 duration/time symbol 固定為 us。Recipe 是 ordered Formula Segment list，預設一段；總長度永遠是各段 duration 總和，insert/delete segment 會改變總長度；`t` 是 segment-local time，`T` 是 whole-waveform global time；segment render 使用半開區間且最後一段包含終點；segment duration 不需對齊 internal render grid；formula `sample_count = round(total_duration * 1000) + 1`，render sample rate 固定為 `1000 samples/us` 且不進 recipe API，並受 `1_000_001` sample 上限約束；第一版不支援 formula 內 conditional；`normalize:"none"` 在 `Abs > 1` 時 fast-fail，`normalize:"peak"` 以 whole-waveform `peak_abs = max(Abs)` 同縮放 I/Q，`peak_abs == 0` 時 no-op，scale factor 即時計算、不持久化、不顯示。
- [0032 — Arbitrary waveform data uses a reference time axis](0032-arbitrary-waveform-reference-time-axis.md)：Arbitrary waveform playback data 帶 reference time axis，不依 ML arb length 做 stretch/compress；`style:"arb"` waveform entry 只存 asset `data` key，playback length 由 asset `time[-1]` 決定。
- [0033 — Arbitrary waveform delete and rename do not scan ModuleLibrary references](0033-arbitrary-waveform-delete-no-reference-scan.md)：刪除/rename Arbitrary Waveform Asset 不掃描也不遷移所有 `ModuleLibrary` waveform/module references；仍引用 missing data 的使用端在 load/playback 時 fail-fast。`update_formula` 會覆寫既有 asset 的 playback arrays 與 embedded recipe。
- [0034 — ArbWaveformDatabase owns shared arbitrary waveform asset operations](0034-arb-waveform-database-shared-asset-repository.md)：`meta_tool.ArbWaveformDatabase` 擁有 shared arbitrary waveform asset repository 規則，GUI/notebook 共用；`list()` 是不開 `.npz` 的 cheap directory index，`inspect()` / preview 才載入單筆 asset 並即時計算 summary；GUI service/controller 只做互動與 remote/MCP adapter policy。
- [0035 — Arbitrary waveform MCP failures use tool errors](0035-arb-waveform-mcp-failures-use-tool-errors.md)：Agent-facing arb waveform MCP tools 沿用 measure-gui 既有錯誤 contract；成功才回 result payload，validation/collision/missing/readiness failure 走 failed tool call，底層 RPC error envelope 帶 stable `reason` 與 optional `data`。`set_arb_waveform` 成功回 `success:true`、`status:"created"|"overwritten"` 與 `preview_figure`，不使用 normal `{success:false,...}` payload；arb asset mutation bump `arb_waveforms` resource version，agent-facing mutation 使用 stale guard。

## VII. 繪圖

- [0016 — notebook liveplot auto_close=True 預設](0016-notebook-liveplot-auto-close-default.md)：不 hack ipympl 私有協議。
- [0017 — worker 線程畫圖：marshal vs 純通知分流](0017-worker-thread-plotting.md)：worker 直接呼 pyplot→marshal；畫圖留主線、worker 只通知→queued signal。

## VIII. autofluxdep

- [0018 — autofluxdep orchestrator 純需求解析器](0018-autofluxdep-orchestrator-requirement-resolver-builder-currying.md)：三介面（requires/provides/produce）+ Builder 柯里化統一 Node 與 Service。
- [0020 — session-core 共用層（gui/session/）](0020-session-core-shared-layer.md)：measure 量測 session core（context/SoC/device/dialog）抽共用、autofluxdep 整套複用；app-local（OperationGate/BackgroundService）vs shared（Handles/Progress/IOManager/QtProgressTransport）邊界 + SessionControllerPort 契約 + Phase B cfg-driven 模擬 run path。

## IX. 多 agent 協作

- [0022 — orchestrator-owned worktree protocol](0022-agent-coordination-worktree.md)：多 agent / 長線 orchestration 使用 `.agent_state/` 下的一 task 一 worktree、state.json、reports 與 plans；orchestrator 建立/指派/驗證/合併，taskboard MCP 不再使用。
- [0023 — Cooperative interrupt：feedback 喚醒 pending wait](0023-cooperative-interrupt-feedback-wakeup.md)：**（已被 [[0025]] 取代）** 原設計給單一共用 await 加 thread-safe feedback inbox 作第二喚醒源；其「多 channel + 時序敏感 combine」形狀會生 race，由 [[0025]] 的單一有序 channel 取代。
- [0024 — Agent launch UI 退役](0024-embedded-agent-session-architecture.md)：**（已退役）** measure-gui 不再提供 toolbar「Agent」按鈕、`AgentLaunchDialog` 或 `services/agent_launcher.py`；agent 啟動責任回到外部 CLI/MCP workflow。runtime `FeedbackPanel` / Send / Send & Stop / `gui_prompt_user` 保留，由 [[0025]] 定義。
