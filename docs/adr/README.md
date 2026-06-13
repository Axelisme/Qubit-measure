# Architecture Decision Records

每篇 ADR 以**現在式**描述目前生效的設計；被取代/推翻的設計只保留簡述 + 推翻原因（折進對應 ADR 的「演化」段，不另立檔）。檔名編號依主題分組排序，非建立時間序。

> 程式碼註解與記憶檔以 `ADR-NNNN` 引用本目錄；ADR 之間以 `[[NNNN]]` 互鏈。本目錄已入 git 追蹤（同 CLAUDE.md「文件追蹤」），會進 diff 與 commit。

## I. 並發、守衛與生命週期

- [0001 — Permit / Lease typed guard](0001-permit-lease-typed-guard.md)：型別強制的 Permit（呼叫前可靜態證明的前置）與 OperationLease（動態 hardware 互斥）分離。
- [0002 — 版本表 + async handle + off-main handler](0002-version-table-async-handle-off-main.md)：並發感知＝資源版本表 guard（非追 origin）+ RPC-as-proxy operation handle + off-main blocking handler + 三層分工。（併入舊 origin-tracking 與 change-buffer 的演化）
- [0003 — 統一 async cancel + ShutdownCoordinator](0003-shutdown-coordinator-and-registry-cancel.md)：Registry 持 stop_event（cancel/poll/await 三動詞齊備）+ Qt-free 輪詢關閉。（§一「綁死」被 [[0019]] 取代）
- [0019 — Operation = token + opt-in facets + 可插 execution strategy](0019-operation-facets-and-execution-strategy.md)：Exclusion/Handle/Progress/Cancel 四 facet 任意組合 + strategy（OffMain-thread/pool、Main-thread-user-paced、Blocking）多型；拆 Handle 出 gate、抽 BackgroundService、「async in main thread」。取代 [[0003]] §一綁死。

## II. Service 架構

- [0004 — Service 互依的三問規則](0004-service-dependency-three-questions.md)：Query/Command/Reaction 三問取代分層表治循環；含「狀態放 State vs service 私有」兩軸。
- [0005 — Service 角色規範（DDD + Hexagonal）](0005-service-roles-ddd-hexagonal.md)：Driving Adapter / App Service / Aggregate Root / Repository / Driven Adapter 五角色 + 重構結果。
- [0006 — ml/md 內容寫入的唯一權威 = ContextService](0006-single-ml-md-write-authority.md)：經窄 write port，lowering+register 收進 ContextService。
- [0007 — Device 狀態下放 State（SSOT）](0007-device-state-to-state-ssot.md)：DeviceService 退化 driver/worker；persistence 為 State 投影。
- [0021 — Event 所有權：domain module 擁有 enum + payload](0021-event-ownership-domain-modules.md)：domain module 擁有 enum+payload 定義；app 在 bus/EVENT_SERIALIZERS 層組裝；bus 維持 payload-type-key；port 集中各層 ports.py（掛 [[0004]]/[[0005]]）。

## III. CfgEditor session

- [0008 — CfgEditor session](0008-cfg-editor-session.md)：service-owned headless LiveModel + 可插拔 widget viewer。（併入 headless-only / delegated 的演化）

## IV. Spec / Value 設定模型

- [0009 — Spec/Value fluent + LiteralSpec 鎖定 + 每角色 default factory](0009-spec-value-fluent-and-literal-lock.md)
- [0010 — Value 樹永遠完整 + None 統一表「空」](0010-value-tree-complete-none-for-empty.md)：（併入 DisabledRefValue marker 的演化）
- [0011 — CfgSchema.validate 成品邊界靜態檢查](0011-cfgschema-validate-boundary.md)
- [0012 — CfgBuilder value 層組裝 builder](0012-cfgbuilder-value-layer-fluent-assembly.md)

## V. Remote / 傳輸

- [0013 — RemoteControlAdapter 作為第二個 View](0013-remote-adapter-as-second-view.md)：與 MainWindow 平級的 driving adapter + 診斷 off-bus fan-out + ViewProtocol 三拆。
- [0014 — GUI 三 app 共用純傳輸層](0014-gui-shared-transport-layer.md)：NdjsonRpcEndpoint + McpBridge 共用；policy 留各 app。

## VI. 持久化

- [0015 — PersistenceCaretaker（Memento + Caretaker）](0015-persistence-caretaker-memento-single-file.md)：單檔 app-state、關閉才寫。

## VII. 繪圖

- [0016 — notebook liveplot auto_close=True 預設](0016-notebook-liveplot-auto-close-default.md)：不 hack ipympl 私有協議。
- [0017 — worker 線程畫圖：marshal vs 純通知分流](0017-worker-thread-plotting.md)：worker 直接呼 pyplot→marshal；畫圖留主線、worker 只通知→queued signal。

## VIII. autofluxdep

- [0018 — autofluxdep orchestrator 純需求解析器](0018-autofluxdep-orchestrator-requirement-resolver-builder-currying.md)：三介面（requires/provides/produce）+ Builder 柯里化統一 Node 與 Service。
- [0020 — session-core 共用層（gui/session/）](0020-session-core-shared-layer.md)：measure 量測 session core（context/SoC/device/dialog）抽共用、autofluxdep 整套複用；app-local（OperationGate/BackgroundService）vs shared（Handles/Progress/IOManager/QtProgressTransport）邊界 + SessionControllerPort 契約 + Phase B cfg-driven 模擬 run path。

## IX. 多 agent 協作

- [0022 — taskboard 為主協調層、worktree 為輔](0022-agent-coordination-taskboard.md)：多 agent 共享 checkout 的協調＝stdio MCP taskboard（file-backed JSON+flock、path 衝突偵測、read/write 鎖、資源 token、pending/wait、TTL 自動回收、md 視圖）；worktree 解不了 singleton 資源爭用故僅為輔。
- [0023 — Cooperative interrupt：feedback 喚醒 pending wait](0023-cooperative-interrupt-feedback-wakeup.md)：訂閱認證下讓 user 即時糾正操作 GUI 的 agent＝給單一共用 await（`await_outcome`）加第二喚醒源（thread-safe feedback inbox），pending wait 提早返回帶判別 payload `{reason,result?,feedback?}`；擴展 [[0002]]、具體化擱置的 gui-inbox（piggyback 降為被動 residual）。
