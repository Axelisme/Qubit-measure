---
status: accepted
---

# ADR-0067 — GUI 應用核心與前端邊界

**狀態：** accepted。

**Last updated:** 2026-10-20，獨立 SoC dialog 與窄 control facets；predictor 與 experiment figure area 的控件 owner；腳本 session 所有權、動態資源與儀器 adapter 邊界

## 問題與決策

Measure 有 Qt 視窗與 remote 入口，autofluxdep 也複用量測 session。把業務狀態交給各前端管理，會使相同操作走不同的驗證和寫入路徑。本篇約束共享能力的 owner、前端反應及繪圖責任，不規定各 app 的 run loop 或 RPC 格式。

GUI 與 remote 是 application 的 driving adapters。它們讀同一個 owner 公開的狀態或投影，寫入時呼叫 owning command，不另造一份可提交的業務狀態。這不表示四個 GUI 都開放相同的 remote commands：autofluxdep、dispersive 的 remote 入口目前只讀；fluxdep 與 Measure 的 remote 呼叫同一 owning commands。Measure 的 `Controller` 和 control facets 提供共用的應用入口；畫面預覽、canvas、焦點與 renderer 仍由 Qt 前端持有。Remote 的傳輸、診斷投遞及 wire 契約留給 Remote／Transport，不由本篇指定。

## Session、狀態與協作

`gui.session` 供 measure 和 autofluxdep 共用 context、SoC、device、操作機制與 dialog 所需的窄 control facets。兩個 app 各自組裝 `SessionServices`，注入背景執行與 operation gate；app 決定自己的 RUN policy、workflow 和呈現。

`SessionServices.runner` 與 `SessionServices.progress` 保留 build 輸入的同一 instances。同 session 的 Measure Run、Analyze、PostAnalyze 與 operation control 從 bundle 取用，不另傳一份 infrastructure。Composition root 仍建立並注入基礎物件，operation policy、exclusion 和 owner-thread delivery 不變。Session 不反向 import 具體 app。

SoC dialog 透過獨立 `SocControlPort` 提供連線與 soccfg 瀏覽，Setup 透過 `SetupControlPort` 管理 entry 與 working points；使用者於 2026-10-20 裁定：「我規劃將soc連線與顯示的邏輯從setup dialog獨立出來，成為獨立的dialog並有獨立入口按鈕，新setup dialog的working points移到原本soc區域的右側。」

Setup 等 shared dialog 只依賴窄 control facet 表達一般能力（開啟 entry、選取 point、讀取記住的 Session Preference、連線），不知道 app 正在啟動、還原或關閉；control port 也不提供啟動專用的版本。Session Preference 只回答下次輸入預填什麼，不證明 entry 已開啟或儀器已連線。Fluxdep、dispersive 並未因此取得完整的 measurement session。

可被多方讀取的 session 與 app 事實由 State 或其 owner 公開讀取投影。ContextService 私下持有 ResultEntry、point 與 library handles，是 GUI 參數與 library 的寫入權威。EMPTY 的 parameters 是 None。未開 entry 時沒有 entry metadata 或 source binding；開啟 entry 後仍為 EMPTY，不自選 point，但可讀 entry metadata、point list 與 setup source binding。選取或建立 point 後才使用 ACTIVE 的持久文件。SoC、device 與獨立 predictor 不因沒有參數而停用。SavedPoint 是唯一參數寫入目標，寫入經 resources owner 的 DocumentStore 單檔交易，成功後才發布目前 binding 的固定投影、版本與 fact。Parameter／library 寫入要求目前已有選取的 point。同 entry 的非選取 point 可寫入但不改目前 selection；跨 entry 拒絕。Schema、來源、衝突或 I/O 失敗不發布候選。

Measure 的 library editor 交未 lower 的 schema 與固定 ParameterContext。ContextService 透過 app 注入的 callbacks 準備 ModuleSpec／WaveformSpec，再提交一筆 library structure change。Rename 不級聯修改其他引用。

WritebackService 先以分析來源的固定 capture 解析與 lower 全部選定 proposals，再依提案順序交 concrete items 給 `ContextService.apply_writeback`。此入口驗證固定 SavedPoint、綁定資源並交 resources 共用 executor。未選工作點時，application 在 append 前沿 no_context 拒絕，提示先選或建立工作點。同 entry 選取其他 point 只允許命令開始，不改 captured SavedPoint 的寫入目的地；跨 entry 仍拒絕。Executor 追加 accepted 後逐項提交，首錯停止，保留成功前綴與事件，不提供跨檔 rollback 或 retry。Context owner 發布實際成功前綴。GUI 標記該前綴 applied 後原樣拋 producer cause，保留既有 wire 分類。Parameter 的 metadata 整份替換為 `{stderr: x}`，沒有 stderr 時為 `{}`，不寫 source、kind、run_id 或時間。Module 的 stderr 隨 accepted provenance 保存。Parameter／Module／Sample proposal、Form draft 與 selected Apply 的契約見 [[0008]]、[[0065]]；子模組搬遷不改成功前綴、權限、busy admission、欄位驗證或錯誤分類。

目前 point 的有序 alias 表由 ContextService 擁有。元件與 device 共用同表，每列保存名稱或引用前列的 `$` 軟連結，不做 kind 或名稱可用性預驗。Caller 以 `alias_presets` 提供具名 catalog，default rows 僅作為未開 entry 的 memory 初值。框架不宣告領域 preset，也不反向 import zcu_lab。

普通 `set_alias_rows` 只改 memory，不要求 point。Dialog Apply、RPC 與 startup Apply 要求已選 point，否則以 `no_context` 拒絕。Apply 先原子驗證整表，再由 entry owner 保存該 point 的 `points/<label>/aliases.yaml`，成功後才安裝 memory 與發布 fact。失敗保舊表、舊檔與 dialog draft。開 entry 或 clear 清空表，選取 point 就換成該 point 的表。缺檔視為空，不回退 setup 或 preset，也不自填空列。這不寫 preference、setup alias、其他 point 容器或 ledger；保存分界見 [[0063]]。

使用者於 2026-10-20 裁定：「採用乙，alias. yaml的獨立檔案，找不到就視內容為空，與其他兩個資料容器並列」。Setup 與 point 的表獨立，建立時複製，之後不繼承。

公開 `read_context(aliases=None)` 固定 point、library、storage 與全表，不檢查 consumer kind。GUI adapter、Tab、Run、Load 和 template create 直接使用它，不宣告角色。裸 `$alias` 回名稱，欄位路徑延後交讀取 owner 檢查。非空具名 overrides 只改本次 capture，不保存到 tab、全域表或檔案。EMPTY 無覆寫由 app 保留 None 的 context-free／deferred 路徑；非空覆寫拒絕 no_context。Notebook 保留自己的顯式 role kind 契約與 caller 提供的 bindings。

Shared `gui.session.ui.alias_panel` 是 Measure 與 workflows 的 Inspect Aliases 編輯入口，不設 app-local wrapper。Alias draft 固定載入時的 entry／point，clean 時跟隨切換，dirty 時保留並拒絕跨 target Apply。Revert 或重開載入目前 point，避免舊草稿靜默覆寫另一個 point。同 target Apply 保持 Inspect 開啟，Cancel 不撤回已 Apply 的表。Tab 不顯示角色列，建立時固定目前 point／library、alias 表、cached device infos 與 value sources。Refresh 重新 capture 並保留 authored inputs，Reset 重新 capture 並重算 defaults。來源事件不自動刷新，Run 使用目前 tab 的有效 Form result 與固定來源。FIT、interactive、Post、Writeback 和 Save 使用結果的固定來源，不因後來選取而換目的地。Native Load 的來源與保存規則見 [[0063]]，cfg acceptance 見 [[0065]]。

Device 的連線資訊、記得但未連線列與 cached DeviceInfo 在 `SessionState`，`DeviceService` 控制 driver 與 in-flight 工作。記哪些 device、重啟時嘗試哪些連線由腳本決定，framework 不從 domain session 快取回填。Setup preferences 只是 live 預填值。State 是否供多方觀察與資料是否重啟後保存是兩個問題，worker、lease 和 driver 的控制權不因有人需要觀察而交給 State。

Measure framework 只保存 Qt 外觀，不保存或還原 SoC、devices、entry／point、tabs／input，也不提供 legacy session 遷移。預設腳本的重開與 on_close 保存見 [[0064]]。Setup 的 aliases.yaml 路徑維持不變，每個 point 另有自己的表。預設腳本不另存或恢復 alias，恢復 point 就自然載入。Workspace 的 RAM 快照只供目前程序重建與 reload，不是持久 session schema。

RunContext 提供 devices 與 cancel_signal 等執行能力，不提供儀器設定方法。Device 層只提供 current／voltage、frequency／power、output 或 value 等操作，不知道 flux／RF 角色。使用者套件 helper 負責把角色用途與明示單位對應到 driver 操作；新增儀器接口不因此擴充 RunContext。這沿既有 C8 分層與 C9 無循環 contracts，experiment 使用 device，而非由 device 反向理解實驗。

`zcu_lab.v2._support.device.check_flux_unit(cfg, unit)` 只比對有效 Info 的 `value_unit`，不讀寫硬體或 cache。YOKO Info 擁有 mode／unit validator，Fake 的單位為 native；其他 Info 類型報 TypeError。`set_flux` 委派同一檢查，保既有 mode 查詢、driver safety、stop 與 progress，不新增 get_info 查詢。

Workflow 宿主提供同步 `get_device_cfg` 與完整 cfg 的 `setup_device`，zcu_lab 工具組合讀取、單位檢查與修改。框架不新增 value＋unit effect 或部分 cfg 合併。

一般實驗與 Workflow 都不在 Run 前預查 device 名稱。Core 實際取 `context.devices[name]` 時，缺名即報錯並指名 device。Workflow 移除 `DevicePort.setup`，不增加 `validate_names`、`names()` 或一般實驗的 `validate_device_names` 預查。Workflow 實際操作 unknown／disconnected device 時，由宿主在該次操作報錯並指名 device，Run 以 failed 停止。這不取消 cfg 格式或必填 unit 的驗證。

State 的 cached DeviceInfo 仍供觀察與 cfg defaults 使用，不是 Run 的控制命令或保存 snapshot。GUI Run 框架在 hardware lease 後、adapter.run 前另讀一次全部已連線 device，任一讀取失敗拒絕 Run。這個 acquisition capture 不更新 GUI State。實驗不對未使用的 device 套用舊 setup patch，固定 snapshot 的保存見 [[0063]]。

Run 結束後，host 另在背景讀取一次已連線儀器，由 State owner 更新 cached DeviceInfo。這只更新觀察值，不改寫 run 前固定 snapshot 或已保存 provenance，也不把 cache 變成下一次 Run 的控制命令。

Device 活動狀態由 driver 在 `device_operation` 中記錄，只有是否在操作中與最後結束的 monotonic 時間。GUI 主 thread 定時讀取 `DeviceManager.activity_snapshot()`，初始約每 200 ms。Device dialog 負責標記與時序，初始在操作中或結束後約 1 秒保持標記。不注入 sink，不投遞活動事件，也不提供事件歷史。活動快照、單名稱 registry 與讀回的完整分工見 [[0076]]。

State 提交由單一 owner loop 序列化。Session 的 `OwnerScheduler` 可由 Qt queued signal 或 headless queue 實作；這個規則不等同於要求 core 依賴 Qt 主線程。操作的取消、handle 與 terminal settle 由 Operation owner 決定，本篇不複製生命週期規則。

建立一條依賴時，先辨認互動：查詢走可用的狀態或 read contract，命令呼叫負責該行為的 owner，對已提交變化的反應訂閱 domain fact。Command 可以是單向的 owning-service 合作；不把「service 絕不互調」當成禁令，也不為每條邊建立 Protocol。窄 port 用在跨 owner 或 shared／app 邊界有隔離需求的地方。Port 和 EventBus 不會自動消除循環，仍要檢查誰發起、誰寫入、誰反應。組裝決定注入哪些模組；有界流程的 coordinator 可決定觸發、順序與分支，但仍透過 owner 的 commands／queries 工作，不直接取得它的寫入權威。Startup／mock setup 與 pane 反應是不同流程，不合成萬用 coordinator。

## 動態資源與儀器 adapters

`SoCConnectionService` 在 owner thread install／uninstall board/config，走 SOC_CONNECT gate，Run 或 connect 中拒絕。Idle 時可替換，uninstall 發布 soc=None。`is_mock` 由 SoC 自述，缺旗標時為 False，GUI 只轉發，不藉此選擇模擬模式。SoC 仍由 caller 擁有，service 不假設或呼叫通用 close 方法。

`DeviceService.register_device(name, driver)` 接受已建好的 BaseDevice，讀取 info 並成功登錄後才移轉 GUI close 責任與發布 cache。Live 同名、Run、同名 mutation 或 info／registry 失敗不移轉候選 driver，caller 仍須清理。Idle MEMORY_ONLY row 可替換，保留 remember。這些操作沿用既有 rename、同 address 與單 driver 單名稱規則，見 [[0076]]。

正常關閉先 drain 工作，再反序執行一次 script callbacks，最後嘗試全部 GUI 管理的 drivers。成功列轉 MEMORY_ONLY，失敗列保留 CONNECTED 與 error。DeviceService 全部嘗試後可 raise 彙整錯誤，Measure teardown 記 log 並繼續退出，不自動重試失敗 driver。關閉接線見 [[0064]]、[[0066]]。

`gui.session.devices` 擁有明確登錄的儀器 adapter catalog。每個儀器 module 同檔放 adapter 與既有 Qt panel。Base 定義 `BaseDeviceAdapter`、`DeviceCapabilities(bias_source, user_connectable)` 與 panel protocol；adapter 宣告 driver_cls、panel_cls、capabilities 並提供 `connect(address)`。Core driver／Info 不變，DeviceService 與 DeviceDialog 共用 catalog，不各養一套 type registry。新增儀器需要 driver、GUI module 與一行明確登錄，不自動掃描目錄。

FakeDevice、YOKO、SGS 維持專屬面板，不改寫成 Form。未知 type 顯示空白頁，不新增通用或唯讀 info panel，也沒有預設 factory。`bias_source` 管 cached bias／unit 能力。`user_connectable=False` 只禁止 GUI 手動 Add；腳本 register／remember 的既有已知 row 仍能經 adapter 在原 address Reconnect，改名後相同。它不是所有連線的禁令，也不依模擬身份特判。

使用者於 2026-10-19 要求：「我認為service層儘量與具體儀器類別解耦」「我認為應該參考core->adapter->service的設計，由一個歸屬於gui層的adapter來描述device」。同日對 adapter 與 panel 同檔、明確登錄及搬移現有面板的方案回答：「可以」。因此儀器知識集中在 GUI adapter，service 只使用其契約。

## 獨立 predictor

Predictor 不持有 installed session model，不參與 Run 或 Analyze／Post-analyze。
`simulate` 擁有框架內建 fluxonium／transmon 的具名參數與中性 query 契約。
`resources.predictor_file` 擁有 `zcu.predictor` JSON 讀寫與驗證，檔案放在 entry 的 `predictors/`。
Session predictor service 透過 ContextControlPort 取得 ResultEntry 擁有的目錄，集中列檔、檔名驗證與載入。
Dialog／remote 使用檔名或明確參數，不解讀 component 欄位，也不由 alias 選來源。
Framework 不 import 使用者套件（C16）；simulate 不讀容器或檔案（C8、C9）。
Dialog 擁有可手動修改的獨立 trial，不寫回元件或來源檔案。Service／remote 不讀 dialog。
Calibrate 只計算候選 bias。Simulation coordinator 只組裝 SoC 與 devices。

使用者於 2026-10-20 裁定：「框架內支援的fluxonium/transmon是理論上的兩種qubit，有自己定義的參數，與用戶定義的component解偶」。
檔案交接採用原話：「我認為還有一個最簡單的，不轉。讓fluxdep的工具寫preditor要用的格式檔，而不是component的欄位。」
對 entry 層檔案、格式與 dialog 方案，使用者回答：「1先entry吧。2可以。3可以。」

2026-10-19，使用者裁決：「採 B，只限 PredictorDialog。依 repo 規則，新 class 有理由：它們各自隱藏真實的控件與局部 state 協調（參數編輯、曲線互動），不是轉接層。照你列的硬邊界（PredictorControlPort／ModelParams／物理 owner 不變、public 行為與文字版面不變、無 Protocol/factory/mixin、callers/tests 同票）。外觀不得改變；ComponentDriver 情境＋逐步 PNG 你親看即可，不需交使用者。」

`PredictorParameterEditor` 持有 model 選擇、nullable 參數與已知欄位控件，以及顯式 file 預填。`PredictorCurveTabs` 持有三張 canvas、toolbars、matrix capability 與跨圖 marker／highlight／xlim 協調。兩者是 dialog-owned 的具體 child QWidget，不取得整個 dialog，也不跨 owner 存取私有狀態。
Dialog 保留 cache、device 訂閱、calibration、tracked table 與生命週期。子元件沿用原 predictor queries 與物理參數型別。Programmatic 預填以同步 callback 保留立即刷新、保存與例外傳遞；不以 Qt signal 隔離取代同步呼叫。這個責任劃分只適用 PredictorDialog。

## Experiment figure area

Measure 的 `ExpTabWidget` 持有具體 child QWidget `ExperimentFigureArea`。Child 集中 Run、Analysis 與 Post 的 canvas hosts、兩個具名圖 selectors 與 cache、interactive presentation、Data gallery 和右側頁切換。它隱藏切圖時的 signal blocking、具名 collection identity、前景預覽刷新及 canvas cleanup 次序。清除 canvas 不銷毀 caller 保留的 Figure。

Root 保留左側表單與 ledger、動作、訂閱、progress、splitter 寬度與 State 投影。既有 ledger 與其 builders 留在同一 ui package。Root 的公開 figure/container 方法委派給 child。Data gallery 經 typed callback 呼叫 root 當時的公開 figure-reader，保留覆寫語意及逐 pane 失敗隔離。Data 啟用先刷新、再切頁、最後 reflow。Child 不取得整個 tab、Controller 或 State，root 不保留搬走控件的 aliases，也不跨 owner 讀 private state。Controller、State、TabActions、capabilities、Save、Writeback 與 Progress 的責任不變。這是同一 Qt frontend 內的持有關係，不新增跨 package 架構邊界。

`ExperimentFigureArea` 是具體控件 owner，不引入 Protocol、factory、mixin、registry 或額外轉接層。原公開行為、同步例外、外觀與時序保持不變。Run、Analyze、Load、admission 與持久化規則仍由既有 owner 負責。MainWindow close-flow 不變。這個分工不改依賴、設定或跨 package 邊界。

## Adapter capability 與事件

Measure 的 `artifacts` package 擁有 ArtifactTracker、SaveService、SaveControl、保存中心與 ArtifactHandover。Run result 是 adapter-owned opaque handle；RunService 不以 RunRecord 型別或 cfg／result 欄位推測其內容，acquired 沿固定 metadata 寫入，不保存 cfg。

Adapter 的 data_item 與 analysis_item 只交框架固定內容，不選格式、路徑或寫檔。Figure 取已提交 pane 的具名原生圖。框架在 owner thread 同步編碼，資料準備失敗交共用 experiment.artifacts 報出對應 data outputs，其他圖與分析仍嘗試保存。

LoadService 擁有 native 讀檔與 tag 比對，再呼叫 adapter.restore(stored)。BaseAdapter 委派核心 from_stored；框架只讀 RestoredRun 的 opaque record 與 cfg_issue，不探測歷史 cfg。cfg 不可用仍是成功 Load，reply 的 cfg_issue 是原因字串或 null，不再提供 has_cfg_snapshot。讀檔、restore 或歷史來源建立失敗時保留舊成果。Load 不回填表單。原始 cfg 保存在 native 資料檔，透過公開 datafile.load_run_data 離線讀取為 StoredRun，GUI 不提供檢視入口。使用者裁定、metadata、分析來源與 storage 分界見 [[0063]]，typed restore 見 [[0069]]。

使用者於 2026-10-14 裁定：「擴充空間，那這兩個功能的定位應該是gui框架下的一個子模塊，對外解耦，對內可擴充。」並釐清：「Artifact目前，一個Artifact是一個寫入單位，但內部支援多個不同的項目。」對 adapter 只交內容、框架決定格式回答：「好，同意做法一」。

GUI handover 與 Qt-free experiment 核心分層，依 GUI → experiment → datafile／resources 的 C8、C9 contracts；共用核心另由 artifacts-core-no-ui 禁止 GUI／Qt 依賴。2026-10-15 以「1可以，2採用忽略不支援欄位的方式，3同意，4一起做吧」確認 Load 不回填與兩個子模組一併實作。

Measure 的 `writeback` package 擁有 Parameter／Module／Sample proposals、opaque draft、
service、control 與 Qt widget。Adapter 公開入口再匯出 proposal 型別；resources.entry
executor 與 remote handler／wire types 留在原 owner。`artifacts` 與 `writeback` 不直接
互相 import，由 app composition 協調，仍可共用 State、events 與 context。
`.importlinter` 的 `measure-artifacts-no-writeback` 與 `measure-writeback-no-artifacts`
contracts 分別檢查兩個方向。

Framework 擁有它消費的實驗 adapter interface，實驗側提供宣告和行為，app composition 注入 catalog。`AdapterCapabilities` 明示 SoC 需求、analysis、post-analysis 與 load 的支援範圍。Measure 的 Qt tab 依 analysis／post-analysis 宣告建立控制項；remote 的 writeback subtab params 依宣告拒絕不支援的分析類別。Run guard 依 `requires_soc` 檢查連線；`GuardService.check_load` 與 LoadService 依 `load_data` 拒絕不支援的 load，不要求目前 context。這些局部檢查不代表所有 application 操作入口都已用同一份宣告拒絕不支援的操作。`check_analyze` 只要求 Run result，不要求目前 context，也不檢查 analysis capability。非 interactive 的 analyze 會進 FIT 路徑，post-analyze 入口也未查 post-analysis capability。補齊這些入口的核准目標與轉正條件見 [GUI capability draft](draft/gui-adapter-capability-guards.md)。

Capability 不等於當次 readiness、檔案相容、權限或 hardware lease。Framework 在既有 guard 和 operation 邊界分別判斷部分 context／cfg 與動態 busy 狀態，run guard 呼叫 adapter 的 preflight；preflight 不代替 execution guard 或硬體操作。精確 flags、conditional hooks 和 import-time 驗證見 [measure adapter contract](../../lib/zcu_tools/gui/app/measure/README.md) 及 [experiment adapter owner](../../zcu_lab/v2/gui-contracts.md)。

Domain 模組定義事件 enum、payload 和已提交的 fact。Producer 不傳 widget 名稱、刷新旗標或重畫遮罩。App 組裝 bus 訂閱與對外投影；GUI coordinator 把 fact 轉為畫面動作。例如 measure 的 tab content fact 在完整 pane state commit 後發布，operation terminal fact 與 content commit 是不同事件，避免成功時畫兩次。具體的保留 figure 恢復順序由 [measure app](../../lib/zcu_tools/gui/app/measure/README.md) 管理。Remote 以自己的投影呈現相同事實；這不規定 wire payload 或診斷通道。

共用 `gui.interactive` 擁有 Qt-free `Session`、typed `Action`、`Command` 與 `PluginDefinition`。App service 持有已提交的 plugin session，Action 對最新 snapshot 驗證並一次提交。Session 提供單層 undo，消耗上一個成功 commit 前的 snapshot，不提供 redo。`.importlinter` 的 `interactive-below-apps` contract（C17）允許 measure／fluxdep app 依賴共用 interactive，禁止共用 interactive 反向依賴 app policy。Qt frontend 可維持尚未提交的 pointer preview，不能把它當作另一份分析狀態。GUI 與 remote command 走同一份 plugin Action；session 終結與 operation settle 由 app owner 處理。`tab.interact` 的純讀與 command 都要求 RPC holder，見 [[0070]]，並保留 active-session、command 與 terminal 驗證。已接受的 command 在 owner loop 依提交順序生效，沒有 last-operator 身分或 plugin revision。MCP 固定 `tab_interact` tool 只轉送一個 read 或 command；read 不切焦點，經驗證的 command 切到 Analysis pane，回傳 PNG 由 MCP 以當次記憶體 image content 交付，不建立 session 暫存 PNG 檔案。點選、預覽、失敗復原和 widget cleanup 見 [measure app](../../lib/zcu_tools/gui/app/measure/README.md)。

## 繪圖與限制

GUI 透過 operation-owned Plots 與明確的 QtPlotHost 使用原生 Figure。Qt frontend 擁有 canvas、container 與 presentation 的生命週期；release 不銷毀 caller 保留的 Figure。通用 [liveplot](../../lib/zcu_tools/plotting/liveplot/README.md) 的 standalone backend 契約仍服務其 public plotters，GUI 不透過該契約註冊自訂 pyplot routing。

Worker 可回傳資料，由主執行緒繪圖，也可透過 explicit host 的 owner scheduler 更新 live artists。兩者都不依賴 ambient pyplot routing。Mathtext parsing 保留 lock／prewarm；這不保證任意 Matplotlib 計算 thread-safe。Queued signal 也不會把共用的可變 Result 變成不可變 snapshot。各 app 仍決定資料的讀寫時機與所有權，具體 presentation 契約見 [GUI plotting](../../lib/zcu_tools/gui/plotting/README.md)。

## Source catalog reload

使用者在 2026-10-09 裁定：「A」。採用的分工是將 source-loading 抽到 shared GUI loader，各 app 提供 typed catalog builder。Source inventory 與 import 機制可共用，但 Measure Registry 與 workflow catalog 的宣告不同，由 app 決定如何建立與驗證。這樣 workflows 不必複製 loader，Measure 也保留原 recovery／UI。

`gui.catalog_loader.SourceCatalogLoader[CatalogT]` 擁有 bounded source inventory、prepare/load、fixed-source restart，以及共用 SourcePackage、PreparedCatalogReload 和 CatalogReloadError。App 提供 typed catalog builder，負責宣告 validation、active admission、發布和 recovery；shared loader 不依賴 measure Registry 或具體 workflow。Measure 保留原 loader 簽名的 registration adapter與原 recovery/UI。

Workflows 的 app-owned catalog 配對目前 native step、plan／knobs form models、converters 和 seed，並由附件建立 panels。Registry 僅驗宣告與名字，不新增 lookup。Owner-thread reload 在 pending/running/paused 時拒絕，不進入 prepare/import。成功保留 compatible parsed drafts、替換附件，不重建 session/services/controller/hardware。失敗保留 RAM draft 與舊 panels，但 Start 必須等成功 retry。Fixed source 改變要求重啟。Bounded import 可能已有 Python 副作用，不宣稱 rollback，也不以舊 catalog fallback 啟動。

新 Start 使用目前附件，在固定 capture 中轉換 forms，建立 detached native plan、tunables、seed 與 exact run roots，再提交給同一 Controller。Controller 持有該 run 的 tunables binding 與固定 Start capture，reload 不從新 catalog 或目前 draft 重建它。Saved run 的轉換依 [[0062]] 由獨立工具讀取 manifest 和 iteration records，GUI／Controller 不保留 export binding。這些接線不新增 RPC listing、wire schema 或 Registry lookup。

## 取捨與相鄰文件

不維護 service tier 表，也不把所有呼叫塞進 Controller。前者無法說明單條邊的用途，後者會放大依賴及模糊寫入 owner。Frontend 可以有不同 presentation，卻不能擁有另一份可提交的業務真相。

Cfg 的編輯和 lowering 邊界見 [[0065]]；持久化見 [[0063]]；workflow 見 [[0062]]；process composition 見 [[0064]]；operation owner-loop、取消與 settle 見 [[0066]]；scheduler 與 gate presence 的局部機制見 [[0053]]。Remote／Transport 的協定見 [[0068]]。本篇不宣稱四個 app 具有相同能力或所有可變繪圖資料已有跨線程 snapshot 保證。
