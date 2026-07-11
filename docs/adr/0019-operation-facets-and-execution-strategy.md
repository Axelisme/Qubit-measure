---
status: accepted
---

# Operation = token + opt-in facets + 可插 execution strategy（拆 Handle 出 gate、抽 BackgroundRunner）

**狀態：** accepted（facet 拆分仍生效；OffMain 細節已演進為 `BackgroundRunner` + caller-owned ambient thunk，`OperationRunner` policy/spec 見 [[0026]]）。
**關聯：** 重構 [[0002]]（gate = `_OperationExclusion` + `_OperationRegistry` facade、共用 token、analyze handle-only）——本檔把 registry 從 facade 拆成正交 sibling。**取代** [[0003]] §一「生命週期綁死、不拆可選疊加」的決定（其「無不互斥長任務場景」前提已失效）；cancel 三動詞承 [[0003]]。strategy 選擇留 domain service，依 [[0004]]/[[0005]]（orchestrator 組合 leaf）。OffMain 畫圖 marshal 依 [[0017]]；guard 用 Permit 依 [[0001]]；`BackgroundExecutor` scope wiring 依 [[0026]]。

## 脈絡

並發 operation 機制長到現在有兩處張力：

1. **四份各自獨立的 off-main 實作**：`RunWorker` / `AnalyzeWorker` / `SaveDataWorker`（三個近乎同形的 QThread + 三個 runner），加上 interactive widget 自己的 `QThreadPool`。每個 worker 把 ambient scope（figure routing / liveplot backend / pbar factory / cancel）**硬編**進 `run()`，cancel 判讀也散在 worker。

2. **gate 把 exclusion 與 handle 綁死,但兩者消費者集合不同**。[[0002]] 的 `OperationGate` 是 `_OperationExclusion`（硬體互斥）+ `_OperationRegistry`（poll/await/cancel handle）的 facade,共用一 token。但 **FIT / INTERACTIVE analyze 要 handle 不要 exclusion** —— code 自招（`OperationKind.ANALYZE` 註解):「takes a lease **only for the async handle**, not for exclusion」。[[0003]] 當時以「無具體不互斥長任務場景」否決拆分;該前提現已失效。

關鍵洞察:**「async」是相對 caller/agent,不是相對 thread**。INTERACTIVE analysis 跑在主線程、由用戶 paced,但對 agent 仍是個 pending 到用戶按 Done 才 settle 的 job —— 這就是「async in the main thread」。一旦看穿,device / run / analysis / interactive / save 全是同一個概念:**開一個 job,不論跑 main / off-main,支援查看進度、中斷、取消**。

## 決策

### 一、統一概念:Operation = token + opt-in facets + 一個 execution strategy

一個 operation 由一個 token 識別,周圍掛**要才綁**的 facet,並由**恰一個** execution strategy 驅動。

**facets(皆 keyed by 同一 token):**

| facet | 語義 | 用者 |
| --- | --- | --- |
| **Exclusion** | 硬體互斥,能不能現在起（conflict matrix） | run / device |
| **Handle** | poll / await / terminal outcome（= operation_id 本體） | run / analyze / interactive / device |
| **Progress** | 可觀測進度（token-keyed `ProgressService`,既有） | run / device-setup |
| **Cancel** | 中斷+取消（request → 驅動者自行 interrupt,持 stop_event） | run / device /（interactive 可選） |

**execution strategy(唯一隨 op 變的軸 —— 誰把 Handle 推向 settle):**

- **OffMain-thread**（bg 專屬 QThread）—— run / FIT analyze /（device 可選）
- **OffMain-pool**（bg 共享 QThreadPool）—— auto-align 等短 helper
- **Main-thread-user-paced** —— INTERACTIVE analyze:Handle 的 settle trigger 是**用戶 Done**,非 worker return（即「async in main thread」）
- **Blocking / event-driven** —— device connect（無 stop 點）

**對映表(同一組 facet 任意組合,非每個都有全部):**

| op | Exclusion | Handle | Progress | Cancel | strategy |
| --- | :--: | :--: | :--: | :--: | --- |
| run | ✅ | ✅ | ✅ | ✅ | OffMain-thread |
| FIT analyze | — | ✅ | — | — | OffMain-thread |
| INTERACTIVE analyze | — | ✅ | — | (可) | **Main-thread-user-paced** |
| device setup | ✅ | ✅ | ✅ | (可) | Blocking / OffMain |
| save | — | — | — | — | OffMain（fire-forget） |
| auto-align | — | — | — | — | OffMain-pool（fire-forget） |

**facet opt-in 是防 god-object 的關鍵**:save / auto-align 只有 strategy、沒 Handle/Progress/Cancel,不被逼長出用不到的東西。「每個 op 都一樣」指**同一組 facet 可任意組合**,不是「每個都有全部」。

### 二、Handle / lifecycle 從 gate 拆成正交 sibling（取代 [[0003]] §一綁死）

- `OperationGate` 瘦成**純 Exclusion**;`_OperationRegistry`(Handle + Cancel,含 stop_event 與 cancel/poll/await 三動詞)升為**獨立 sibling collaborator**;Progress 已是 token-keyed sibling,不動。
- **analyze / interactive 改「只拿 Handle、不拿 lease」**;run / device 拿 Exclusion + Handle。消除「借 lease 當 handle」的假象。
- 終端流程解耦:domain work（writeback / State）→ **settle Handle** → **release Exclusion**(若有)。今 `gate.release` 合一的兩步拆成兩個獨立呼叫。

### 三、BackgroundRunner = OffMain strategy 的乾淨機制

- 單一入口 `submit(work, *, run_in_pool, on_done, on_error)`。`run_in_pool` 直接講機制:True = 共享 QThreadPool（短 helper）、False = 專屬 QThread（長 operation）—— 避開 starvation（長 op 不搶 pool）。
- Ambient scope 不再由 bg 物件辨識；caller 在 `work` thunk 內組合所需 scope（[[0026]]）：
  - `figure_ambient` 住 app 層，封裝 routing ContextVar 與 QtLivePlotBackend。
  - `progress_ambient` 住 session 層，只承載 pbar ContextVar。
  - run policy closure 內把 `stop_event` 轉成 `schedule_stop_scope(StopSignal(...))`。
- `RunWorker` / `AnalyzeWorker` / `SaveDataWorker` + 三 runner **收斂成一個 generic worker + bg**；scope 下沉成 service 建好的 thunk；**cancel 判讀上移 domain service**（持 stop_event 者在 terminal policy 自判 finished vs cancelled+partial，worker 只回 done/failed）。
- interactive widget 經**窄 `InteractiveHostEnv` port**(目前只含 `run_background`,由 ctrl 實作)用 bg 的 pool strategy;**不持整個 ctrl**(command-surface 才持 ctrl,passive host 收窄注入,測試塞 fake port)。

### 四、刻意不做（防過度抽象）

- **不建正式 `ExecutionStrategy` class 階層**:strategy 選擇留 domain service(本就是知情 orchestrator),抽 Strategy 物件對 ~5 個 call site 是 over-abstraction(同 [[0003]] 防過度設計精神)。
- **gate 不 wrap bg**:Exclusion / Handle / Execution 三正交,service 組合,互不巢狀。
- **不強制所有 op 有 Handle**:save / auto-align 只有 strategy。

## 替代方案與否決理由

- **gate 作 bg 的使用方,外包 Lease/Lock**:把正交三軸綁成巢狀;interactive(Exclusion❌ Handle✅) 與 save/auto-align(Exclusion❌ Execution✅) 證明正交;且終端 domain 邏輯(writeback/State)必須留 service,gate 驅動 bg 會逼出 gate→service 反向回呼。否決 → 三 sibling 由 service 組合。
- **正式 ExecutionStrategy 多型階層**:domain service 已知情,Strategy 物件徒增 indirection。否決。
- **interactive widget 持整個 ctrl**(類比「View 持 ctrl」):寬依賴滿足窄需求,7 個 widget 測試要 ctrl,bg.submit API 漏進 View widget;且該 widget 今天**對 ctrl 下零命令**(拖線/action→session、Done→注入 callback),是 passive host 非 command surface。否決 → 窄 `InteractiveHostEnv` port。
- **liveplot backend 與 routing 拆成兩個 scope**:co-dependent —— `QtLivePlotBackend.make_plot_frame`→`plt.subplots`→`require_current_container()`,沒 routing 直接 crash;analyze 只設 routing 不設 liveplot,只因它從不呼 liveplot API(no-op),非需要解耦。否決 → 同一 `figure_container` facet。
- **bg 大一統吃 main-thread work**:main / off-main 是不同 strategy,硬塞一個「什麼都 Optional」god-worker。否決 → strategy 多型、Handle 統一。

## 演化（取代 [[0003]]）

[[0003]] §一刻意「生命週期綁死、不拆 exclusion/handle 可選疊加」,並在「考慮過的替代」否決拆分,理由「用戶接受綁死、無具體不互斥長任務場景、過度設計」。**該前提現已失效**:FIT 與 INTERACTIVE analyze 正是 handle-not-exclusion 的具體場景(code 已自招借 lease 當 handle)。本檔據此拆成 opt-in facets。

保留自 [[0003]]:cancel 三動詞(request → worker 自判 cancelled,持純 stop_event 零 callback)移入 Handle leaf 不變;`ShutdownCoordinator` 輪詢關閉(§三)不變 —— 它對 token `poll`/`cancel`,只是 token 的 handle 來源從 gate-registry 改為獨立 Handle leaf。

## 落地（皆完成）

- **Phase A（GUI 22 / Phase 146）** —— 抽 OffMain strategy + `InteractiveHostEnv` port；後續由 [[0026]] 收斂為 `BackgroundRunner.submit(work, *, run_in_pool, on_done, on_error)`，ambient scope 由 caller thunk 擁有。三 worker/三 runner（`runner.py`）收斂成一個 generic worker；cancel 判讀上移 RunService。範圍 run / analyze / save / interactive。
- **Phase B（GUI 23 / Phase 147）** —— `OperationGate` 瘦成純 Exclusion（`ensure_can_start`/`register`/`release`）;新 `operation_handles.OperationHandles`（`create` mint token、`settle`/`await_outcome`/`poll`/`cancel`/`cancel_all`/`live_count`）成 sibling;analyze/interactive 改**只拿 Handle 不拿 lease**;run/device/connect 組合兩 leaf;`ShutdownCoordinator`/`QtShutdownDriver`/`Controller.await_operation` 改吃 `OperationHandles`;`active_operation_count = handles.live_count`（現含 analyze/interactive）。
- **device execution 統一（GUI 24 / Phase 148）** —— device 的 `_DeviceCommandWorker`/`_DeviceSetupWorker` 兩個 QThread 也收斂進 `bg.submit`（connect/disconnect 無 scope；setup 帶 progress scope，stop_event 由 work closure 捕捉、driver 直接 poll，**非** experiment schedule scope；cancel 判讀進 `DeviceService._on_setup_done`，「setup 可不可取消」改讀 `_active_kind`）。至此 run/analyze/save/interactive/device 全走同一個 OffMain strategy 機制。
- 三階段 WIRE 皆不變、皆全綠驗證。

## 範圍

`gui/app/main/`(services / runner / ui)+ `experiment/v2_gui/adapters/`(`_support/interactive_flux_pick` 只呼 `host` port,不變)。Phase B 另涉 `device` / `connection` / `services/remote` dispatch。**adapter 與 wire 契約不變**。
