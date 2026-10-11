# ADR-0067 — GUI 應用核心與前端邊界

**狀態：** accepted。

## 問題與決策

Measure 有 Qt 視窗與 remote 入口，autofluxdep 也複用量測 session。把業務狀態交給各前端管理，會使相同操作走不同的驗證和寫入路徑。本篇約束共享能力的 owner、前端反應及繪圖責任，不規定各 app 的 run loop 或 RPC 格式。

GUI 與 remote 是 application 的 driving adapters。它們讀同一個 owner 公開的狀態或投影，寫入時呼叫 owning command，不另造一份可提交的業務狀態。這不表示四個 GUI 都開放相同的 remote commands：autofluxdep、dispersive 的 remote 入口目前只讀；fluxdep 與 Measure 的 remote 呼叫同一 owning commands。Measure 的 `Controller` 和 control facets 提供共用的應用入口；畫面預覽、canvas、焦點與 renderer 仍由 Qt 前端持有。Remote 的傳輸、診斷投遞及 wire 契約留給 Remote／Transport，不由本篇指定。

## Session、狀態與協作

`gui.session` 供 measure 和 autofluxdep 共用 context、SoC、device、操作機制與 dialog 所需的窄 control facets。兩個 app 各自組裝 `SessionServices`，注入背景執行與 operation gate；app 決定自己的 RUN policy、workflow 和呈現。

`SessionServices.runner` 與 `SessionServices.progress` 保留 build 輸入的同一 instances。同 session 的 Measure Run、Analyze、PostAnalyze 與 operation control 從 bundle 取用，不另傳一份 infrastructure。Composition root 仍建立並注入基礎物件，operation policy、exclusion 和 owner-thread delivery 不變。Session 不反向 import 具體 app。

Setup 等 shared dialog 只依賴窄 control facet 表達一般能力（開啟 entry、選取 point、讀取記住的 Session Preference、連線），不知道 app 正在啟動、還原或關閉；control port 也不提供啟動專用的版本。Session Preference 只回答下次輸入預填什麼，不證明 entry 已開啟或儀器已連線。Fluxdep、dispersive 並未因此取得完整的 measurement session。

可被多方讀取的 session 與 app 事實由 State 或其 owner 公開讀取投影。ContextService 私下持有 ResultEntry、point 與 library handles，是 GUI 參數與 library 的寫入權威。EMPTY 沒有參數；開啟 entry 後建立記憶體 DRAFT，選取 point 後使用 ACTIVE 的持久文件。SavedPoint 寫入經 resources owner 的 DocumentStore 單檔交易，成功後才發布目前 binding 的固定投影、版本與 fact。DRAFT 只提交記憶體候選。同 entry 的非選取 point 可寫入但不改目前 selection；跨 entry 和失效 DRAFT 拒絕。Schema、來源、衝突或 I/O 失敗不發布候選。

Measure 的 library editor 交未 lower 的 schema 與固定 ParameterContext。ContextService 透過 app 注入的 callbacks 準備 ModuleSpec／WaveformSpec，再提交一筆 library structure change。Rename 不級聯修改其他引用。WritebackService 則先以分析來源的固定 capture 解析全部選定的 `$alias.path` proposals，再追加 accepted，依序呼叫 ContextService 的 parameter／module／waveform setters。首個 setter 失敗停止，保留已成功寫入與 accepted 證據，不宣稱跨檔原子性或 rollback。

全域有序 alias 表由 ContextService 擁有。元件與 device 共用同表，每列保存名稱或引用前列的 `$` 軟連結，不做 kind 或名稱可用性預驗。Caller 以 `alias_preset` 提供 Measure 初值，框架不宣告領域 preset，也不反向 import zcu_lab。使用者套件的唯一初值是 `ALIAS_PRESET`，flux_dev 空列不猜裝置或 cache。

普通 `set_alias_rows` 只改 memory。Dialog Apply 先原子驗證整表，再由 entry owner 保存 `aliases.yaml`，成功後才安裝 memory 與發布 fact。EMPTY 只改 memory，失敗保舊表、舊檔與 draft。每次開 entry 有檔就載入，無檔重設 caller preset 並重新啟用第一份非空 point 的 primary 唯一候選初始化。Saved 空表與空列不自填，切 point 不重讀或換表。這不寫 preference、point、setup 或 ledger；保存分界見 [[0063]]。

公開 `read_context(aliases=None)` 固定 point、library、storage 與全表，不檢查 consumer kind。GUI adapter、Tab、Run、Load 和 template create 直接使用它，不宣告角色。裸 `$alias` 回名稱，欄位路徑延後交讀取 owner 檢查。非空具名 overrides 只改本次 capture，不保存到 tab、全域表或檔案。EMPTY 無覆寫由 app 保留 None 的 context-free／deferred 路徑；非空覆寫拒絕 no_context。Notebook 保留自己的顯式 role kind 契約與 caller 提供的 bindings。

Shared `gui.session.ui.alias_dialog` 是 Measure 與 workflows 的同一編輯入口，兩者直接 import，不設 app-local wrapper。Apply 保持 dialog 開啟，Cancel 不撤回已 Apply 的表。Tab 不顯示角色列，也不在建立時固定環境。Cfg refresh 使用目前 point／library 和 alias 表；Run admission 才固定 ParameterContext、解析名稱與 storage capability。FIT、interactive、Post、Writeback 和 Save 使用結果的固定來源，不因後來選取而換目的地。Native Load 的來源與保存規則見 [[0063]]，cfg acceptance 見 [[0065]]。

Device 的可觀察狀態在 `SessionState`，`DeviceService` 控制 driver 與 in-flight 工作；remembered device 的磁碟保存由 Persistence owner 從已提交狀態選擇性投影。State 是否供多方觀察與資料是否重啟後保存是兩個問題。Live SoC handle 可屬 session environment 而不落盤，worker、lease 和 driver 的控制權也不因有人需要觀察其狀態而交給 State。

RunContext 提供 devices 與 cancel_signal 等執行能力，不提供儀器設定方法。Device 層只提供 current／voltage、frequency／power、output 或 value 等操作，不知道 flux／RF 角色。實驗層 helper 負責把角色用途與明示單位對應到 driver 操作；新增儀器接口不因此擴充 RunContext。這沿既有 C8 分層與 C9 無循環 contracts，experiment 使用 device，而非由 device 反向理解實驗。

`experiment.utils.check_flux_unit(cfg, unit)` 只比對有效 Info 的 `value_unit`，不讀寫硬體或 cache。YOKO Info 擁有 mode／unit validator，Fake 的單位為 native；其他 Info 類型報 TypeError。`set_flux` 委派同一檢查，保既有 mode 查詢、driver safety、stop 與 progress，不新增 get_info 查詢。

Workflow 宿主提供同步 `get_device_cfg` 與完整 cfg 的 `setup_device`，zcu_lab 工具組合讀取、單位檢查與修改。框架不新增 value＋unit effect 或部分 cfg 合併。

一般實驗與 Workflow 都不在 Run 前預查 device 名稱。Core 實際取 `context.devices[name]` 時，缺名即報錯並指名 device。Workflow 移除 `DevicePort.setup`，不增加 `validate_names`、`names()` 或一般實驗的 `validate_device_names` 預查。Workflow 實際操作 unknown／disconnected device 時，由宿主在該次操作報錯並指名 device，Run 以 failed 停止。這不取消 cfg 格式或必填 unit 的驗證。

State 的 cached DeviceInfo 仍供觀察與 cfg defaults 使用，不是 Run 的控制命令或保存 snapshot。GUI Run 框架在 hardware lease 後、adapter.run 前另讀一次全部已連線 device，任一讀取失敗拒絕 Run。實驗不對未使用的 device 套用舊 setup patch，固定 snapshot 的保存見 [[0063]]。

State 提交由單一 owner loop 序列化。Session 的 `OwnerScheduler` 可由 Qt queued signal 或 headless queue 實作；這個規則不等同於要求 core 依賴 Qt 主線程。操作的取消、handle 與 terminal settle 由 Operation owner 決定，本篇不複製生命週期規則。

建立一條依賴時，先辨認互動：查詢走可用的狀態或 read contract，命令呼叫負責該行為的 owner，對已提交變化的反應訂閱 domain fact。Command 可以是單向的 owning-service 合作；不把「service 絕不互調」當成禁令，也不為每條邊建立 Protocol。窄 port 用在跨 owner 或 shared／app 邊界有隔離需求的地方。Port 和 EventBus 不會自動消除循環，仍要檢查誰發起、誰寫入、誰反應。組裝決定注入哪些模組；有界流程的 coordinator 可決定觸發、順序與分支，但仍透過 owner 的 commands／queries 工作，不直接取得它的寫入權威。Startup／mock setup 與 pane 反應是不同流程，不合成萬用 coordinator。

## Adapter capability 與事件

Framework 擁有它消費的實驗 adapter interface，實驗側提供宣告和行為，app composition 注入 catalog。`AdapterCapabilities` 明示 SoC 需求、analysis、post-analysis 與 load 的支援範圍。Measure 的 Qt tab 依 analysis／post-analysis 宣告建立控制項；remote 的 writeback subtab params 依宣告拒絕不支援的分析類別。Run guard 依 `requires_soc` 檢查連線；`GuardService.check_load` 與 LoadService 依 `load_data` 拒絕不支援的 load，不要求目前 context。這些局部檢查不代表所有 application 操作入口都已用同一份宣告拒絕不支援的操作。`check_analyze` 只要求 Run result，不要求目前 context，也不檢查 analysis capability。非 interactive 的 analyze 會進 FIT 路徑，post-analyze 入口也未查 post-analysis capability。補齊這些入口的核准目標與轉正條件見 [GUI capability draft](draft/gui-adapter-capability-guards.md)。

Capability 不等於當次 readiness、檔案相容、權限或 hardware lease。Framework 在既有 guard 和 operation 邊界分別判斷部分 context／cfg 與動態 busy 狀態，run guard 呼叫 adapter 的 preflight；preflight 不代替 execution guard 或硬體操作。精確 flags、conditional hooks 和 import-time 驗證見 [measure adapter contract](../../lib/zcu_tools/gui/app/measure/README.md) 及 [experiment adapter owner](../../zcu_lab/v2/gui-contracts.md)。

Domain 模組定義事件 enum、payload 和已提交的 fact。Producer 不傳 widget 名稱、刷新旗標或重畫遮罩。App 組裝 bus 訂閱與對外投影；GUI coordinator 把 fact 轉為畫面動作。例如 measure 的 tab content fact 在完整 pane state commit 後發布，operation terminal fact 與 content commit 是不同事件，避免成功時畫兩次。具體的保留 figure 恢復順序由 [measure app](../../lib/zcu_tools/gui/app/measure/README.md) 管理。Remote 以自己的投影呈現相同事實；這不規定 wire payload 或診斷通道。

共用 `gui.interactive` 擁有 Qt-free `Session`、typed `Action`、`Command` 與 `PluginDefinition`。App service 持有已提交的 plugin session，Action 對最新 snapshot 驗證並一次提交。Session 提供單層 undo，消耗上一個成功 commit 前的 snapshot，不提供 redo。`.importlinter` 的 `interactive-below-apps` contract（C17）允許 measure／fluxdep app 依賴共用 interactive，禁止共用 interactive 反向依賴 app policy。Qt frontend 可維持尚未提交的 pointer preview，不能把它當作另一份分析狀態。GUI 與 remote command 走同一份 plugin Action；session 終結與 operation settle 由 app owner 處理。`tab.interact` 保留 active-session、command 與 terminal 驗證，但不要求 per-connection seen；GUI 與 agent 在 owner loop 依提交順序生效，後提交者勝出，沒有 last-operator 身分或 plugin revision。MCP 固定 `tab_interact` tool 只轉送一個 read 或 command；read 不切焦點，經驗證的 command 切到 Analysis pane，回傳 PNG 由 MCP 以當次記憶體 image content 交付，不建立 session 暫存 PNG 檔案。點選、預覽、失敗復原和 widget cleanup 見 [measure app](../../lib/zcu_tools/gui/app/measure/README.md)。

## 繪圖與限制

GUI 透過 operation-owned Plots 與明確的 QtPlotHost 使用原生 Figure。Qt frontend 擁有 canvas、container 與 presentation 的生命週期；release 不銷毀 caller 保留的 Figure。通用 [liveplot](../../lib/zcu_tools/plotting/liveplot/README.md) 的 standalone backend 契約仍服務其 public plotters，GUI 不透過該契約註冊自訂 pyplot routing。

Worker 可回傳資料，由主執行緒繪圖，也可透過 explicit host 的 owner scheduler 更新 live artists。兩者都不依賴 ambient pyplot routing。Mathtext parsing 保留 lock／prewarm；這不保證任意 Matplotlib 計算 thread-safe。Queued signal 也不會把共用的可變 Result 變成不可變 snapshot。各 app 仍決定資料的讀寫時機與所有權，具體 presentation 契約見 [GUI plotting](../../lib/zcu_tools/gui/plotting/README.md)。

## 取捨與相鄰文件

不維護 service tier 表，也不把所有呼叫塞進 Controller。前者無法說明單條邊的用途，後者會放大依賴及模糊寫入 owner。Frontend 可以有不同 presentation，卻不能擁有另一份可提交的業務真相。

Cfg 的編輯和 lowering 邊界見 [[0065]]；持久化見 [[0063]]；workflow 見 [[0062]]；process composition 見 [[0064]]；operation owner-loop、取消與 settle 見 [[0066]]；scheduler 與 gate presence 的局部機制見 [[0053]]。Remote／Transport 的協定見 [[0068]]。本篇不宣稱四個 app 具有相同能力或所有可變繪圖資料已有跨線程 snapshot 保證。
