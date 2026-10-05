# ADR-0067 — GUI 應用核心與前端邊界

**狀態：** accepted。

## 問題與決策

Measure 有 Qt 視窗與 remote 入口，autofluxdep 也複用量測 session。把業務狀態交給各前端管理，會使相同操作走不同的驗證和寫入路徑。本篇約束共享能力的 owner、前端反應及繪圖責任，不規定各 app 的 run loop 或 RPC 格式。

GUI 與 remote 是 application 的 driving adapters。它們讀同一個 owner 公開的狀態或投影，寫入時呼叫 owning command，不另造一份可提交的業務狀態。這不表示四個 GUI 都開放相同的 remote commands：autofluxdep、fluxdep、dispersive 的 remote 入口目前只讀。Measure 的 `Controller` 和 control facets 提供共用的應用入口；畫面預覽、canvas、焦點與 renderer 仍由 Qt 前端持有。Remote 的傳輸、診斷投遞及 wire 契約留給 Remote／Transport，不由本篇指定。

## Session、狀態與協作

`gui.session` 供 measure 和 autofluxdep 共用 context、SoC、device、操作機制與 dialog 所需的窄 control facets。兩個 app 各自組裝 `SessionServices`，注入背景執行與 operation gate；app 決定自己的 RUN policy、workflow 和呈現。Session 不反向 import 具體 app。Setup 等 shared dialog 只依賴窄 control facet 表達一般能力（套用 project、讀取記住的 Session Preference、連線），不知道 app 正在啟動、還原或關閉；control port 也不提供啟動專用的版本。Session Preference 只回答下次輸入預填什麼，不證明 project 已套用或儀器已連線。Fluxdep、dispersive 並未因此取得完整的 measurement session。

可被多方讀取的 session 與 app 事實由 State 或其 owner 公開讀取投影；寫入交給對應 owner。成功提交後，owner 依操作發布版本或變更通知。`ContextService` 是 md／ml 內容的寫入權威，measure 的 cfg lowering policy 經注入的 callback 配合這條寫入路徑，Writeback 不直接改 md／ml。`apply_ml_writes()` 先在 memory-only md／ml 候選完成全部 lowering 與 register；後項使用前項候選結果，準備失敗不修改 live content。成功後保留 stores 的物件身份與路徑，一次安裝完整內容、bump `context` version 並 emit 變更事件，再執行既有同步／dump。儲存失敗明確回報設定已套用但儲存失敗，不回滾已提交內容，也不新增重試、未保存狀態管理或跨檔交易。Selected Apply 的版本接受與其他尚待整合義務見 [Cfg 編輯 draft](draft/cfg-editing-boundaries.md#observationrun-與-apply)。

Device 的可觀察狀態在 `SessionState`，`DeviceService` 控制 driver 與 in-flight 工作；remembered device 的磁碟保存由 Persistence owner 從已提交狀態選擇性投影。State 是否供多方觀察與資料是否重啟後保存是兩個問題。Live SoC handle 可屬 session environment 而不落盤，worker、lease 和 driver 的控制權也不因有人需要觀察其狀態而交給 State。

State 提交由單一 owner loop 序列化。Session 的 `OwnerScheduler` 可由 Qt queued signal 或 headless queue 實作；這個規則不等同於要求 core 依賴 Qt 主線程。操作的取消、handle 與 terminal settle 由 Operation owner 決定，本篇不複製生命週期規則。

建立一條依賴時，先辨認互動：查詢走可用的狀態或 read contract，命令呼叫負責該行為的 owner，對已提交變化的反應訂閱 domain fact。Command 可以是單向的 owning-service 合作；不把「service 絕不互調」當成禁令，也不為每條邊建立 Protocol。窄 port 用在跨 owner 或 shared／app 邊界有隔離需求的地方。Port 和 EventBus 不會自動消除循環，仍要檢查誰發起、誰寫入、誰反應。組裝決定注入哪些模組；有界流程的 coordinator 可決定觸發、順序與分支，但仍透過 owner 的 commands／queries 工作，不直接取得它的寫入權威。Startup／mock setup 與 pane 反應是不同流程，不合成萬用 coordinator。

## Adapter capability 與事件

Framework 擁有它消費的實驗 adapter interface，實驗側提供宣告和行為，app composition 注入 catalog。`AdapterCapabilities` 明示 SoC 需求、analysis、post-analysis 與 load 的支援範圍。Measure 的 Qt tab 依 analysis／post-analysis 宣告建立控制項；remote 的 writeback subtab params 依宣告拒絕不支援的分析類別。Run guard 依 `requires_soc` 檢查連線；load permit 與 LoadService 依 `load_data` 拒絕不支援的 load。這些局部檢查不代表所有 application 操作入口都已用同一份宣告拒絕不支援的操作。Analyze permit 目前只查 context／run result，非 interactive 的 analyze 會進 FIT 路徑，post-analyze 入口也未查 post-analysis capability。補齊這些入口的核准目標與轉正條件見 [GUI capability draft](draft/gui-adapter-capability-guards.md)。

Capability 不等於當次 readiness、檔案相容、權限或 hardware lease。Framework 在既有 guard 和 operation 邊界分別判斷部分 context／cfg 與動態 busy 狀態，run guard 呼叫 adapter 的 preflight；preflight 不代替 execution guard 或硬體操作。精確 flags、conditional hooks 和 import-time 驗證見 [measure adapter contract](../../lib/zcu_tools/gui/app/measure/README.md) 及 [experiment adapter owner](../../zcu_lab/v2/gui-contracts.md)。

Domain 模組定義事件 enum、payload 和已提交的 fact。Producer 不傳 widget 名稱、刷新旗標或重畫遮罩。App 組裝 bus 訂閱與對外投影；GUI coordinator 把 fact 轉為畫面動作。例如 measure 的 tab content fact 在完整 pane state commit 後發布，operation terminal fact 與 content commit 是不同事件，避免成功時畫兩次。具體的保留 figure 恢復順序由 [measure app](../../lib/zcu_tools/gui/app/measure/README.md) 管理。Remote 以自己的投影呈現相同事實；這不規定 wire payload 或診斷通道。

共用 `gui.interactive` 擁有 Qt-free `Session`、typed `Action`、`Command` 與 `PluginDefinition`。App service 持有已提交的 plugin session，Action 對最新 snapshot 驗證並一次提交。Session 提供單層 undo，消耗上一個成功 commit 前的 snapshot，不提供 redo。`.importlinter` 的 `interactive-below-apps` contract（C17）允許 measure／fluxdep app 依賴共用 interactive，禁止共用 interactive 反向依賴 app policy。Qt frontend 可維持尚未提交的 pointer preview，不能把它當作另一份分析狀態。GUI 與 remote command 走同一份 plugin Action；session 終結與 operation settle 由 app owner 處理。`tab.interact` 保留 active-session、command 與 terminal 驗證，但不要求 per-connection seen；GUI 與 agent 在 owner loop 依提交順序生效，後提交者勝出，沒有 last-operator 身分或 plugin revision。MCP 固定 `tab_interact` tool 只轉送一個 read 或 command；read 不切焦點，經驗證的 command 切到 Analysis pane，回傳 PNG 由 MCP 放到 session 專屬暫存。點選、預覽、失敗復原和 widget cleanup 見 [measure app](../../lib/zcu_tools/gui/app/measure/README.md)。

## 繪圖與限制

GUI 透過 operation-owned Plots 與明確的 QtPlotHost 使用原生 Figure。Qt frontend 擁有 canvas、container 與 presentation 的生命週期；release 不銷毀 caller 保留的 Figure。通用 [liveplot](../../lib/zcu_tools/plotting/liveplot/README.md) 的 standalone backend 契約仍服務其 public plotters，GUI 不透過該契約註冊自訂 pyplot routing。

Worker 可回傳資料，由主執行緒繪圖，也可透過 explicit host 的 owner scheduler 更新 live artists。兩者都不依賴 ambient pyplot routing。Mathtext parsing 保留 lock／prewarm；這不保證任意 Matplotlib 計算 thread-safe。Queued signal 也不會把共用的可變 Result 變成不可變 snapshot。各 app 仍決定資料的讀寫時機與所有權，具體 presentation 契約見 [GUI plotting](../../lib/zcu_tools/gui/plotting/README.md)。

## 取捨與相鄰文件

不維護 service tier 表，也不把所有呼叫塞進 Controller。前者無法說明單條邊的用途，後者會放大依賴及模糊寫入 owner。Frontend 可以有不同 presentation，卻不能擁有另一份可提交的業務真相。

Cfg 的編輯和 lowering 邊界見 [[0065]]；持久化見 [[0063]]；workflow 見 [[0062]]；process composition 見 [[0064]]；operation owner-loop、取消與 settle 見 [[0066]]；scheduler 與 gate presence 的局部機制見 [[0053]]。Remote／Transport 的協定見 [[0068]]。本篇不宣稱四個 app 具有相同能力或所有可變繪圖資料已有跨線程 snapshot 保證。
