---
status: accepted
---

# 0067 — GUI 應用核心與前端邊界

**Last updated:** 2026-11-17，文件組由 session 持有、Entry 視窗命名；2026-10-29，workflow 嵌入 measure 與 headless owner；獨立 SoC dialog 與窄 control ports；predictor 與 experiment figure area 的控件 owner；腳本 session 所有權、動態資源與儀器 adapter 邊界

## 問題與決策

**規則。** Qt GUI 與 remote 都是 application 的 driving adapters。兩者讀 owner 公開的狀態或投影，寫入時呼叫 owning command，不各自保存可提交的業務狀態。Measure 的 experiment 與 workflow 共用 session。

**例子。** Measure 的 GUI 和 remote 透過 `Controller` 與窄 control ports 共用應用入口。畫面預覽、canvas、焦點與 renderer 由 Qt 前端持有。

**邊界。** 共用 owner 不表示開放相同的 remote commands。Dispersive 的 remote 入口目前只讀；Fluxdep 與 Measure 的 remote 各自呼叫與 GUI 相同的 owning commands。本篇不規定各 app 的 run loop。傳輸、診斷投遞與 wire 契約見 [[0068]]。

**依據。** 前端若各自管理業務狀態，相同操作就可能走不同的驗證與寫入路徑。共用入口與各前端分工見 [Measure owners](../../lib/zcu_tools/gui/app/measure/README.md#owners) 及下列各節。

## GUI callback lifetime

**規則。** GUI 元件把 callback、Qt signal、訂閱、timer、event filter 和 matplotlib cid 登記到 `gui.lifetime.LifetimeScope`。可替換的 editor、row 或 frontend 各有自己的 scope。宿主先取得 domain 的關閉許可，再呼叫 `close()`，成功才刪除 Qt children。

**例子。** Editor 的 scope 若先登記訂閱、後登記 timer，`close()` 會先永久停用 callback，再先停 timer、後解除訂閱。清理遇到首個例外就拋出，未完成的項目留待重試，已成功的項目不重做。失敗不會讓 callback 恢復，也不能因此刪除 children。已進入的 callback 不會被中斷，從可重入的 owner command 返回後要先確認 scope 仍開著才更新畫面。

**邊界。** Scope 不關閉 borrowed session、不 drain 工作，也不代替 operation 的取消或終局。外部 Qt parent 刪除 C++ owner 後，晚到的 callback 不做事，但這不能代替明確清理。不要用擷取 owner 的 `destroyed` closure 清理。弱 receiver、FormEditor 的 weak view 與 event filter proxy 見 [Callback lifetime](../../lib/zcu_tools/gui/README.md#callback-lifetime-lifetimepy)。`.importlinter` 的 `gui-lifetime-no-domain` contract 禁止此機制依賴 app 或 domain。

**依據。** 2026-10-25，使用者要求：「这种生命周期的问题非常可怕，尤其是这种停用然后或释放或静态之类的，最好有个架构上就可以解决的方案，简单的方案就可以了。」對統一 scope 答「采用建议」。原裁決未附入庫討論，見 Git `707afb251`。

## Session、狀態與協作

**規則。** `gui.session` 供應 context、SoC、device、共用操作機制與窄 control ports。Composition root 建立並注入基礎物件。Experiment 與 workflow tabs 使用同一份 `SessionServices`，共用 runner、progress、operation gate 與 RUN lease。`AppServices` 只列 app 自己建構的 services，session services 從 `services.session` 取得，Controller 的公開 properties 不變。

State 或 owning service 公開唯讀投影，State 的提交由單一 owner loop 序列化。ContextService 持有 ResultEntry 及已安裝的 global、作用中文件組，是 GUI 參數、library 和 alias 的寫入權威。每組含 parameters.yaml、library.yaml、aliases.yaml。文件組一起切換、一起捕捉；寫入走 resources 的 DocumentStore 單檔交易，成功才發布對應投影、版本與 fact。

合作前先辨認互動。查詢走 State 或 read contract，命令交 owning service，已提交的變化用 domain fact 通知。Owning services 可以單向呼叫；需要跨 owner 或 shared／app 隔離時才用窄 port。Port 和 EventBus 不代替依賴循環檢查。Composition 決定注入，coordinator 只安排有界流程的觸發、順序與分支，不接管 owner 的寫入。Startup／mock setup 和 pane 反應各有自己的流程。

**例子。** 開 entry 即從 EMPTY 進入 GLOBAL，不自選 point。選取或建立 point 時，ContextService 將作用中文件組整組換成 POINT。`use_point(None)` 換回 global，`clear_context` 才關閉 entry。GLOBAL 與 POINT 都可 Run、Analyze、Save 和 writeback，不逐名稱 fallback。SoC、device 及獨立 predictor 各自決定 admission。`DocumentSet` 只讀已安裝的 handles、內容和 revision，不做 I/O 或整組 commit；resources 也不提供整組開檔。文件組 Readiness、`DocumentSetRef`、固定 `DocumentSetCapture`、FileSync 安裝與檔案版本見 [Session README](../../lib/zcu_tools/gui/session/README.md) 和[文件組討論](../discussions/2026-11-17-global-point-parameter-types-and-document-set.md)。同 entry 的非作用中組可寫，不改 selection、作用中版本或舊 capture；跨 entry 拒絕。Schema、來源、衝突或 I/O 失敗不發布候選。

**邊界。** ContextService、PredictorService 直接滿足各自的窄 port，組裝處檢查結構相容，欄位與參數仍用 Port 限制能力。ContextService 的 `current_point_label()` 只讀已安裝的 selection，未選 point 回傳 None，不做 I/O。Device、Entry、SoC 保留有組合或訂閱邏輯的 facets，Progress facet 不變。Shared dialog 只看一般能力，沒有啟動專用 port，不知道 app 在啟動、還原或關閉。SoC dialog 經 `SocControlPort` 連線與瀏覽 soccfg，Entry 視窗經 `EntryControlPort` 管 entry 與 working points。Session Preference 只供預填，不證明 entry 或儀器已開。Session 不決定 workflow 或 tab 呈現，不反向 import 具體 app，也不讓 Fluxdep 或 Dispersive 因此取得完整量測 session。Runner、progress 的 instance 身分與 delivery 見 [Session 的組裝規則](../../lib/zcu_tools/gui/session/README.md#關鍵設計)。

Measure 的 library editor 交未 lower 的 schema 與固定 capture，ContextService 用 app 注入的 callbacks 準備 ModuleSpec／WaveformSpec，再提交一筆 library structure change。Rename 不級聯修改引用。WritebackDraft 用分析來源的固定 capture 求值，再依提案順序交 concrete items。Preview 顯示當下位置；Apply 從當下的 `read_context().target` 取得目的地，由 ContextService 驗證、綁定資源並交共用 executor。未開 entry 可 Preview，非空 Apply 在 append 前拒絕；空歷史環境本身不阻擋寫回。Receipt 的 destination 不含 kind，已提交的 entry_name 是字串。提交順序、版本和 committed index 見 [[0063]]、[Session README](../../lib/zcu_tools/gui/session/README.md) 及 [Writeback Apply](../../lib/zcu_tools/gui/app/measure/writeback/README.md#apply)。GUI 依 committed index 標記 draft item ID，再原樣拋出 producer cause，保留 wire 分類。Parameter metadata 整份替換為 `{stderr: x}`，沒有 stderr 時為 `{}`，不保存來源、kind、run_id 或時間。Module 只保存 Spec，不保存 per-leaf stderr 或來源。Accepted 事件與 Record 的事件引用仍保留。提案、Form draft 與 selected Apply 見 [[0008]]、[[0065]]；分包不改權限、busy admission、欄位驗證、成功前綴或錯誤分類。

元件與 device 共用有序 alias 表。每列可寫名稱或以 `$` 引用前列，不做 kind 或名稱可用性預驗。Global 與 point 的表獨立，建 point 時複製，之後不繼承。兩處缺檔都當空，不建檔、不 fallback；缺參數檔仍是錯誤。`apply_alias_rows` 先驗整表，指定目前 focus label 才寫 point，None 寫 global，保存成功才安裝 rows、revision 並發布 fact。Startup 必須先開 entry，不暫存 rows，也沒有 memory-only rows 或 presets。Framework 不反向 import `zcu_lab`。`read_context` 固定參數、library、storage 和全表；Tab、GUI adapter、Run、Load 與 template create 共用這條讀取路徑，不宣告角色。具名 override 只改當次 capture，裸 `$alias` 讀名稱，欄位路徑由讀取 owner 驗證。EMPTY 無覆寫保留 app 的 context-free／deferred 路徑，非空覆寫拒絕。Notebook 仍可宣告 role kind 並提供 bindings。同步與缺檔衝突見 [[0063]]、[[0081]] 及 [Session environment](../../lib/zcu_tools/gui/session/README.md#contextsession-environment-與-run-snapshot)。

Measure 與 workflows 共用 `gui.session.ui.alias_panel`。Inspect 三頁可選 Active documents 或 Entry template，有效編輯寫入所選位置，New library entry 也固定該位置的 capture。寫入失敗保留輸入和舊檔。同 target 更新保留 dirty draft 和原 revision；切換來源或關閉才丟棄未提交輸入。選了 point 時，entry 的明確寫入更新 cache 和 revision 並發非作用中通知，外部 reload 則只發 file status。Tab 不顯示角色列。來源變化不刷新 tab；Refresh 保留 authored inputs，Reset 重算 defaults，兩者都重新 capture。Run 用 tab 目前有效的 Form 與固定來源；FIT、interactive、Post 和 Writeback 求值用結果的固定來源。Save 以目前 roots 與 Run 記錄的名稱定目的地。其餘來源與 draft 契約見 [Measure 的 sources and workflow](../../lib/zcu_tools/gui/app/measure/README.md#sources-and-workflow)、[[0063]]、[[0065]]。

DeviceService 擁有 driver 與 in-flight 工作；SessionState 的連線、記住的列、cached DeviceInfo 是讀取投影，不是 Run 控制或保存 snapshot。腳本決定記住哪些 device、重啟時連哪些。`DialogPrefill` 只是 live 預填。Measure framework 在磁碟只保存 Qt 外觀，不保存或還原資源、tabs 或輸入，不遷移 legacy session；不另存 alias，開 entry 或選 point 就載入該表。Workspace RAM 快照只供同程序重建和 reload。預設腳本與 on_close 見 [[0064]]。

RunContext 提供 devices、cancel_signal 等能力，不提供儀器設定方法。Device 層只認儀器操作，不認 flux／RF 角色；lab helpers 負責角色及明示單位。`zcu_lab.v2.support.device.check_flux_unit(cfg, unit)` 只比 Info 的 `value_unit`，不讀寫硬體或 cache；YOKO 驗 mode／unit，Fake 用 native，其他 Info 報 TypeError。`set_flux` 共用此檢查，不增 `get_info`，保留既有 mode 查詢、safety、stop 與 progress。Workflow 宿主供同步 `get_device_cfg` 與完整 cfg 的 `setup_device`，不提供 value＋unit effect 或部分 cfg 合併。一般實驗及 workflow 不預查 device 名稱；實際取用才指名拒絕缺名，workflow 的 unknown／disconnected 操作由宿主拒絕並使 Run failed。這不免除 cfg 格式及必填 unit 驗證，也不增加 `DevicePort.setup` 或名稱列舉／驗證接口。Run 不對未使用的 device 套用舊 setup patch。Run 前固定 capture 與 Run 後背景讀回各自讀一次已連線的 devices；前者供保存，不更新 State，後者只更新觀察 cache，不改 snapshot 或 provenance。時機與錯誤語意見 [Session 的 Run snapshot](../../lib/zcu_tools/gui/session/README.md#contextsession-environment-與-run-snapshot)。活動快照只讀 driver 的操作中狀態與最後結束的 monotonic 時間：GUI 主線程初始約每 200 ms 讀一次，Device dialog 在操作中或結束後約 1 秒保留標記，這個時長是初始設定。不投遞活動事件、不注入 sink 或存歷史。見 [[0063]]、[[0076]]。

State 的 owner loop 可用 Qt queued signal 或 headless queue，不要求 core 依賴 Qt 主線程。`pane_commit` 擁有 Measure 的 pane 提交、artifact 規劃、版本、退役清理與未發布 plots 的 discard。State 只交換已備妥的資料並回傳退役資源；唯讀 preview 與 swap 共用退役選擇，不改 State 或清理。Service／control 轉接失效事實，remote 只投影既有 wire 字串與順序。GuardService 判 domain admission，gate 裁 hardware exclusion，operation owner 保留終局、取消及觸發清理的時機。Run、Analyze、PostAnalyze、Load 與 Tab 不各自管理清理。提交順序及失效關係見 [Measure owners](../../lib/zcu_tools/gui/app/measure/README.md#owners)、[[0066]]。這不改 C8／C9 或 artifacts／writeback 分隔契約。

**依據。**
- 2026-10-29，使用者授權評估及選擇窄 port 的實作：「候選8，需要評估trade-off, 候選11可以做，都允許你自己決定」。由 designer 在授權下代裁，讓 owner services 直接滿足窄 port。使用者 2026-11-21 追認：「前两个代材可以」。原始討論未入庫，見 Git `14d72247e`。
- 2026-10-20，使用者裁定：「我規劃將soc連線與顯示的邏輯從setup dialog獨立出來，成為獨立的dialog並有獨立入口按鈕，新setup dialog的working points移到原本soc區域的右側。」原裁決未附入庫討論，見 Git `602ddda77`。2026-11-16，使用者要求「不要再有setup這個名詞與屬性了」，因此視窗與程式識別都改名 Entry。
- 2026-10-29，使用者提出：「沒有激活point就用全域的parameter.yaml/module_cfg/alias，有激活就用point內的，每個都是write-through」。對整組切換答「A，我目前先設計成整個作用域換成point，沒有個別的fallback邏輯」。2026-11-16，對「三檔合稱文件組、由 session 持有、每種檔一個形狀相同的開檔入口」答：「A不錯」。文件組與命名的推導及覆核見 [Global／point 參數型別收斂與文件組](../discussions/2026-11-17-global-point-parameter-types-and-document-set.md)。
- 2026-10-28，使用者說：「我其實沒有要知道來源的需求，這功能純屬於錦上添花」「來源我一直以為只是metadata，沒想要解析」。見 [移除 module provenance 的格式變更](../storage-format-history/2026-10-28-remove-module-provenance.md)。
- 2026-10-20，使用者裁定：「採用乙，alias. yaml的獨立檔案，找不到就視內容為空，與其他兩個資料容器並列」。原裁決未附入庫討論，見 Git `29e550a98`。
- 2026-10-28，使用者答「都可以」，確認 State 預覽與 swap 共用退役選擇，以及 Guard、gate、operation 的分工；對 pane 提交與退役清理集中到一個 module 答「強烈建議的都採用」。原裁決未附入庫討論，見 Git `b73a62268`、`9ca0a18f2`。

## Workflow 嵌入 measure

**規則。** Workflow 嵌入 Measure 的同一程序與 session。每 tab 有自己的 headless workflow owner，持有 Plan、Tunables、固定 form capture、Start lowering、tunables binding 與 run。`WorkflowControlFacet` 管 tab registry 與 native run_id routing。Qt pane 只顯示投影並轉發命令，MCP 直接呼叫同一 facet，不經 pane 或 toggle。`AppServices.session` 與 `Controller.session_services` 公開同一 bundle，不用 bus 訂閱者或私有 runner 找資源。Session 不反向依賴 app（C2）。

**例子。** 一個 toggle 在 Experiment tabs 和 Workflow tabs 間切換，同名 workflow 可多開。等待回答時，tab 選到與 Plan、Tunables 並列的 Ask 子頁；狀態列徽章可切回該 tab。回答或 Stop 後仍保留最近問答。UI、RPC 與 MCP 都讀 owner 留存的最近一題、已接受答案及提交／已回答／取消狀態。Answer 只通知既有 worker。Engine 先比對原題 request_id，再驗答案；錯型答案保持 paused，不會顯示為已回答。問答 envelope 與交付見 [Workflow hosting](../../lib/zcu_tools/gui/app/measure/README.md#workflow-hosting)。

**邊界。** RUN lease 一次只放行一個 experiment 或 workflow。Pending、running、paused 的 workflow tab 不可關閉；pane 取得 owner 同意後才釋放自己的訂閱、timer 和 children。關掉最後一個 pane 也不 disconnect 硬體或 drain session。Coroutine 留在原 worker，不在 UI owner thread 續跑或另排 segment。原 worker 完成 finally 與 executor drain，才由原 owner 交付終局、釋放 lease，見 [[0066]]、[[0075]]。純 figure widgets 留在 UI；native Plots 由 headless owner 管。Pane 和 MCP 共用 workflow／目前 operation 兩群的名稱投影，新 attempt 整組替換 operation plots，workflow 趨勢圖跨 iteration 保留。Terminal 保留最新 operation，replacement Start 和 close 釋放兩組 presentation。名稱、集合和 same-host fork 見 [Workflow hosting](../../lib/zcu_tools/gui/app/measure/README.md#workflow-hosting)、[[0069]]。

**依據。**
- 2026-10-21，使用者裁定：「顯示的tab widget在正常實驗模式就是目前的adapter，在workflow模式下就是每個workflow，你認為如何？」同日確認：「我會希望能支援多個workflow tab」「當然lease會限制只能一次只能跑一個tab/workflow」。承載背景見 [Workflow 定位、承載與執行基底](../discussions/2026-10-21-workflow-positioning-and-hosting.md)，tab 裁決見 Git `2227541f5`。
- 2026-10-22，使用者裁定：「23沒問題，ask請獨立一個subtab而不是取代tunable。再來，ABCD的切換方式都很糟，我只要一個按鈕來toggle即可。」同日的 operation 繪圖裁決與 async host 契約分別見 [[0069]]、[[0075]] 及 Git `935d7f792`。原 tab／Ask 裁決未附入庫討論，見 Git `2227541f5`。

## 動態資源與儀器 adapters

**規則。** `SoCConnectionService` 在 owner thread 替換或卸載 board/config，走 SOC_CONNECT gate，Run 或 connect 中拒絕。卸載發布 soc=None。`DeviceService` 只在讀取 info 且登錄成功後接管 driver 的 close 責任並發布 cache。儀器知識由 `gui.session.devices` 的明確 adapter catalog 持有，DeviceService 與 DeviceDialog 共用。新儀器加 driver、含 adapter 與 Qt panel 的 GUI module 及一筆明確登錄，不掃描目錄。Core driver／Info 不變。

**例子。** 腳本可用 `register_device` 將已建好的 driver 交給 GUI。Idle 的 MEMORY_ONLY row 可替換並保留 remember；live 同名、Run、同名 mutation 或 info／registry 失敗時，GUI 不接管，caller 仍須清理。單名稱與 address 規則見 [[0076]]。`user_connectable=False` 只擋 GUI 手動 Add，不擋已知的既有 row 透過 adapter 在原 address Reconnect，改名後也一樣。

**邊界。** SoC 的 `is_mock` 由 SoC 自述，缺旗標時為 False；GUI 不以此選模擬模式。Caller 提供的 SoC 仍歸 caller，不假設有通用 close；service 自建連線在發布替換或卸載後釋放。正常關閉先 drain、再反序執行 script callbacks 一次、最後嘗試全部 GUI 管理的 drivers，不自動重試。關閉後的 device 狀態與錯誤處理見 [Session README](../../lib/zcu_tools/gui/session/README.md)、[[0064]]、[[0066]]。FakeDevice、YOKO、SGS 用專屬 panel，不改成 Form；未知 type 顯示空白頁，沒有 generic／read-only info panel 或預設 factory。`bias_source` 管 cached bias／unit 能力；連線不以模擬身份特判。`BaseDeviceAdapter` 宣告 driver_cls、panel_cls 與 `DeviceCapabilities`，並提供 `connect(address)`，panel 依 base 的 protocol。見 [Session 結構](../../lib/zcu_tools/gui/session/README.md#結構)。

**依據。** 2026-10-19，使用者要求：「我認為service層儘量與具體儀器類別解耦」「我認為應該參考core->adapter->service的設計，由一個歸屬於gui層的adapter來描述device」。同日對 adapter 和 panel 同檔、明確登錄及搬移現有面板答「可以」。原裁決未附入庫討論，見 Git `a1377ef12`。

## 獨立 predictor

**規則。** Predictor 不持有 installed session model，不參與 Run、Analyze 或 Post-analyze。`simulate` 擁有內建 fluxonium／transmon 的具名參數及中性 query 契約。`resources.predictor_file` 擁有 `zcu.predictor` JSON 的讀寫與驗證；session predictor service 透過 ContextControlPort 取得 ResultEntry 的 `predictors/` 目錄並列檔、驗檔名和載入。Dialog 和 remote 用檔名或明確參數，不讀 component 欄位，也不由 alias 選來源。

**例子。** Dialog 的兩個編輯器共用一份 Qt-free `PredictorTrial`。Trial 擁有手動草稿和試算規則，`PredictorCurveTabs` 持有三張 canvas 與跨圖互動，dialog 協調呈現、live mode、視窗生命週期及 I/O。Calibrate 只算可加到目前 offset 的 Φ0 correction，先呈現 offset 再發布成功 status，不寫來源。請求、控件、預填時序、debounce 與 `PredictorSnapshot` v2 的 cache 契約見 [Session 的獨立 predictor](../../lib/zcu_tools/gui/session/README.md#獨立-predictor)。

**邊界。** 這個控件分工只適用 PredictorDialog，使用者看到的行為、文字與版面不變。Dialog 不把整個 widget 或 owner 私有狀態交給 children。Service／remote 不讀 dialog 或 trial，必須收到所有必要的儀器值與明確 flux frame，不向其他 owner 補值。`analysis.flux` 負責 native 到 reduced flux 的換算；simulation coordinator 只組合 SoC 和 devices。Framework 不 import 使用者套件（C16）；simulate 不讀容器或檔案，GUI → analysis 沿用 C8／C9。Trial 草稿不寫回元件或來源檔；非法草稿不覆寫上次有效的 `predictor.json`，cache 格式不變。

**依據。**
- 2026-10-20，使用者裁定：「框架內支援的fluxonium/transmon是理論上的兩種qubit，有自己定義的參數，與用戶定義的component解偶」。檔案交接：「我認為還有一個最簡單的，不轉。讓fluxdep的工具寫preditor要用的格式檔，而不是component的欄位。」對 entry 層檔案、格式與 dialog 答：「1先entry吧。2可以。3可以。」原裁決未附入庫討論，見 Git `5b14a43e7`。
- 2026-10-19，使用者裁決：「採 B，只限 PredictorDialog。依 repo 規則，新 class 有理由：它們各自隱藏真實的控件與局部 state 協調（參數編輯、曲線互動），不是轉接層。照你列的硬邊界（PredictorControlPort／ModelParams／物理 owner 不變、public 行為與文字版面不變、無 Protocol/factory/mixin、callers/tests 同票）。外觀不得改變；ComponentDriver 情境＋逐步 PNG 你親看即可，不需交使用者。」原裁決未附入庫討論，見 Git `28fe32144`。
- 2026-10-29，使用者授權移動試算狀態與規則：「候選11可以做，都允許你自己決定」。由 designer 在授權下代裁，將試算移到 Qt-free trial。使用者 2026-11-21 追認：「前两个代材可以」。原始討論未入庫，實作見 Git `73e00b1f3` 和上述 Session README。

## Experiment figure area

**規則。** `ExpTabWidget` 持有具體 child QWidget `ExperimentFigureArea`。Child 管右側 Run、Analysis、Post 的 canvas hosts、具名圖的選取及 cache、interactive presentation、Data gallery 和頁面切換。Root 管左側表單與 ledger、動作、訂閱、progress、splitter 寬度和 State 投影。公開的 figure/container 方法委派給 child。

**例子。** 啟用 Data 時，child 透過 typed callback 讀 root 當時的公開 figure-reader，先刷新預覽，再切頁，最後 reflow。覆寫仍生效，單一 pane 讀取失敗不影響其他 pane。清除 canvas 不銷毀 caller 保留的 Figure。切圖與清理契約見 [Measure UI README](../../lib/zcu_tools/gui/app/measure/ui/README.md)。

**邊界。** Child 不取整個 tab、Controller 或 State；root 不保留搬走控件的 aliases，兩者不跨 owner 讀私有狀態。這是 Qt 內部持有關係，不另建 Protocol、factory、mixin、registry 或轉接層。公開行為、同步例外、外觀、時序、MainWindow close-flow 及 app 責任都不變；既有 owner 保留 Run、Analyze、Load、admission 和持久化。既有 ledger 與 builders 留在同一 ui package，不改依賴或設定。

**依據。** 具體 child 將切圖的 signal blocking、collection identity、預覽更新及 canvas 清理留在一起。這是 Measure 內部的控件分工。原變更及其歸屬修正見 Git `02386e6f1`、`ed2afce81`。

## Results browser 唯讀檢視

**規則。** Results browser 是 Measure 選單開啟的獨立頂層視窗。它唯讀索引 Database runs 與 Result ledger，畫面只用保存的圖、metadata、cfg 及 analysis 文件，不問目前 catalog。首次開窗由 app 的索引 owner 借用共用 executor 掃描，逐個 Database entry 在 owner thread 交付。隱藏再開沿用同一份索引，RPC 每次仍完整掃描。

**例子。** 實驗已從 catalog 刪除，仍可用保存的 experiment 字串分組、搜尋或比較。Cfg 顯示 cfg_type、schema_version 及相對欄位路徑；analysis summary 也用相對路徑並另列 parameters。`datafile.read_docs` 同次開檔讀 metadata 與 cfg；壞一份只在該 run 留 issue，不遮住同檔另一份，也不影響其他 runs。查詢與投影見 [Measure owners](../../lib/zcu_tools/gui/app/measure/README.md#owners)。

**邊界。** 瀏覽不回填表單、不重新分析、不寫結果，不遷移 cfg schema，也不取代 Load／Save 的 RunMetadata 驗證。只有按 Open 才由 catalog owner 解析 adapter，沿既有 Load admission 開 tab，成功後選新 tab 並 raise 主視窗。這是 [[0063]] 不新增 cfg 檢視入口的例外，不讓 GUI 探測 adapter 的 live cfg／result。Ledger 由 Result entry owner 唯讀提供，不解析 component、不建立檔案或 lock。依賴沿 C8、C9，framework 不依賴使用者實驗定義。索引不是 operation；光有索引在跑時，關閉主窗不跳 operation 確認，`Controller.close()` 先取消索引再 drain。

**依據。**
- 2026-10-23，使用者裁定：「我傾向與可以在measure-gui中開啟，我考慮的是一個獨立視窗而不是dialog，因為這樣空間比較充足。」同次納入原型的 cfg 身分、版本及相對欄位檢視。原裁決未附入庫討論，見 Git `2bc9107ea`。
- 2026-10-24，使用者裁定：「result browser應該與目前的實驗定義解耦，只是單純的解析cfg與顯示圖就好，就算cfg已經是舊版本version，或是實驗已經刪掉了，應該也不會影響使用。」原裁決未附入庫討論，見 Git `2bc9107ea`。
- 關閉行為由 designer 代裁，使用者 2026-11-19 追認，答「A」。見[開窗與背景索引討論](../discussions/2026-11-18-results-browser-open-speed.md#依據)。

## Adapter capability 與事件

**規則。** Framework 擁有自己消費的 adapter interface，實驗側提供宣告和行為，app composition 注入 catalog。`AdapterCapabilities` 由 Analysis／Interactive 和 PostAnalysis 的定義值推導，不是作者旗標。Adapter 擁有 cfg、Result 及 guide 宣告；Run／restore 用 RunRecord envelope 交接，RunService 只裝入 acquisition metadata、持有並轉交 record。Artifact 保存與 Writeback 各有 owner，不直接互相 import，由 app composition 協調，可共用 State、events 與 context。見 [Measure owners](../../lib/zcu_tools/gui/app/measure/README.md#owners)、[Artifacts](../../lib/zcu_tools/gui/app/measure/artifacts/README.md)、[Writeback](../../lib/zcu_tools/gui/app/measure/writeback/README.md)。

Domain 模組定義事件 enum、payload 及已提交的 fact，不傳 widget 名稱、刷新旗標或重畫遮罩。App 組裝訂閱及投影，GUI coordinator 將 fact 轉為畫面動作，remote 投影同一事實，不由本篇規定 wire payload 或診斷通道。Interactive 的提交走同一份 Qt-free plugin Action，以最新 snapshot 驗證後一次提交。

**例子。** Measure 依 capability 建 Qt 控制項，remote 也依同一宣告拒絕不支援的 writeback subtab。GuardService 先判 catalog reload／shutdown 是否使實驗不可用，再查 action 前提；Analyze／PostAnalyze 在有既有 result 後才檢查能力，最後檢查 tab busy。UI 與 remote 投影同一 admission decision，放行仍不取得 hardware lease。Tab content fact 在完整 pane commit 後發布，與 operation terminal fact 分開，不在成功時重畫兩次。順序與 figure 恢復見 [Tab admission](../../lib/zcu_tools/gui/app/measure/README.md#tab-admission) 和 [Figure presentation](../../lib/zcu_tools/gui/app/measure/README.md#figure-presentation)。

**邊界。** Capability 不等於當次 readiness、相容性或權限。Run 要 SoC 連線，Run、Load 不另要求 storage；Load、Analyze 不要求目前 context。任何 adapter 都可嘗試 Load，相容性由 spec 決定。不相容顯示原因並保留舊成果。實驗額外驗證在 adapter.run 開頭，不另設 `validate_run_request` hook，不代替 execution guard。定義值與建值檢查見 [Measure adapter contract](../../lib/zcu_tools/gui/app/measure/README.md) 及 [實驗 adapter owner](../../zcu_lab/v2/gui-contracts.md)。

Adapter 只交 data_item、analysis_item 與結果內容，不挑格式或路徑，不寫檔。框架在 owner thread 同步編碼，從已提交 pane 取具名原生圖；資料準備失敗由共用 experiment.artifacts 報對應 data outputs，其餘圖和分析仍嘗試保存。RunService 從 `record.metadata` 取得歷史欄位，把 `acquired` 事件追加到 ledger，不在事件中保存 cfg，見 [[0063]]。GUI 不探測 cfg、Result 或分析值的 shape，不收窄 workflow 回傳。GUI handover 與 Qt-free experiment 核心維持 C8／C9 的 GUI → experiment → datafile／resources 方向，`artifacts-core-no-ui` 禁止核心依賴 GUI 或 Qt。`measure-artifacts-no-writeback` 與 `measure-writeback-no-artifacts` 分別保護兩方向；resources.entry executor、remote handler 與 wire types 留原 owner。Pane 持有具名 `WritebackDraft` 或 None，UI 和 remote 不直接操作 draft。Service 協調跨 draft 的預驗證及成功前綴，control 保留 admission、事件和 detached receipt。Draft 狀態與注入的寫入能力見 [Writeback README](../../lib/zcu_tools/gui/app/measure/writeback/README.md)。Proposal 型別由 writeback package 擁有，adapter 公開入口再匯出。

LoadService 讀 native 檔及比對 tag，交 `adapter.restore(stored)`，由 adapter 依 spec 呼叫 `restore_run`。框架只讀 `RestoredRun` 的 envelope、metadata 及 `cfg_issue`。無可用 cfg 仍可成功 Load，回覆的 `cfg_issue` 是原因文字或 null，不另給 `has_cfg_snapshot`。讀檔、restore 或建立歷史來源失敗時保留舊成果。Load 不回填表單；離線可用 `datafile.load` 把 native cfg 讀為 Container，或用 Results browser 唯讀檢視。`open_tab_from_file` 是唯一隔離 `make_form` 失敗的建 tab 路徑，保留明確無 Form 狀態與錯誤，不造空 Form 或回填 cfg。範本綁定、defaults 及 Load 失敗仍傳出；Load 失敗移除新 tab 並還原焦點。無 Form 的 Reset 準備失敗更新原錯呈現並拋出該次例外，有 Form 的 Reset 失敗則保留舊 Form。保存與 typed restore 見 [[0063]]、[[0069]]，Factory 與 Reset 見 [[0073]]。

`Adapter.guide_path` 回傳唯讀的宣告路徑，None 表示未宣告，不讀檔或檢查存在。呈現用 `Adapter.guide()` 讀 Markdown，失敗原樣拋出。

`gui.interactive` 有單層 undo、沒有 redo。App 持有已提交 plugin session 並負責終結和 operation settle，Qt 可有未提交 pointer preview，不另建分析狀態。`.importlinter` 的 `interactive-below-apps`（C17）禁止共用層反向依賴 app policy。`tab.interact` 的讀取與命令都要求 RPC holder，保留 active-session、command、terminal 驗證；accepted commands 按 owner loop 提交順序生效，不存 last-operator 或 plugin revision。MCP 的 `tab_interact` 只轉送一個 read 或 command；read 不切焦點，驗證過的 command 切到 Analysis pane，PNG 以記憶體 image content 回傳，不建 session 暫存檔。見 [[0070]]、[Interactive README](../../lib/zcu_tools/gui/interactive/README.md) 與 [Measure app](../../lib/zcu_tools/gui/app/measure/README.md)。

**依據。**
- 2026-10-14，使用者說：「擴充空間，那這兩個功能的定位應該是gui框架下的一個子模塊，對外解耦，對內可擴充。」並釐清：「Artifact目前，一個Artifact是一個寫入單位，但內部支援多個不同的項目。」對 adapter 只交內容答「好，同意做法一」。2026-10-15，以「1可以，2採用忽略不支援欄位的方式，3同意，4一起做吧」確認 Load 不回填和兩個子模組一起做。原裁決未附入庫討論，見 Git `2976f23d3`。
- 2026-10-22，使用者裁定：「存讀檔接口也不用RunRecord外的資訊了」。同一 envelope 避免 metadata 和資料錯配，不保留 opaque Run fallback。原裁決未附入庫討論，見 Git `727ce8483`。
- 2026-10-24，使用者裁定：「只在 open_tab_from_file 路徑隔離 make_form 失敗，tab 照常建立並 Load，表單區與 Run 顯示 factory 原文錯誤，Reset 走一般 fail-fast；一般 new-tab／Reset 不變」。原裁決未附入庫討論，見 Git `6db415d26`。
- 2026-11-18，使用者對 capability 與 admission 過目：「我大概看了一遍，基本上是没问题」，並說明「并不算是裁决，而算是建议」。這是 agent 的選擇，引用時是僅供參考的先例，不是使用者裁定。原覆核未附入庫討論，見 Git `1f4b16aa0`。

## 繪圖與限制

**規則。** GUI 透過 operation-owned Plots 和明確的 QtPlotHost 使用原生 Figure。Qt frontend 擁有 canvas、container 與 presentation 的生命週期，release 不銷毀 caller 保留的 Figure。不註冊自訂 pyplot routing。

**例子。** Worker 可回傳資料交主線程繪圖，也可經明確的 host owner scheduler 更新 live artists。[`@plots.on_plot_thread`](../../lib/zcu_tools/plotting/README.md#原生-matplotlib-更新) 複製 caller 資料，再請 host owner 執行作者 callback 和 refresh。

**邊界。** 線圖、熱圖及 scatter 工具歸 lab，GUI 不依賴其格式，見 [[0072]]。Mathtext 保留 lock／prewarm，不保證任意 Matplotlib 計算 thread-safe。Queued signal 不會把可變 Result 凍結；各 app 仍決定讀寫時機與資料所有權。呈現契約見 [GUI plotting](../../lib/zcu_tools/gui/plotting/README.md)。

**依據。** 顯式 host 使排程和圖的 owner 不依賴 ambient pyplot routing。移除舊 routing 的紀錄見 Git `69bce6e5b`，live plot 入口見 Git `8ba70ea3b`。

## Source catalog reload

**規則。** Shared `gui.catalog_loader` 擁有有界的 source inventory、prepare/load、fixed-source restart 與共用型別，不認識 Measure Registry 或具體 workflow。App 提供 typed catalog builder，負責驗證宣告、admission、發布和 recovery；Measure 保留既有 registration adapter 簽名與 recovery／UI。API 見 [GUI README](../../lib/zcu_tools/gui/README.md) 及 [Experiment reload](../../lib/zcu_tools/gui/app/measure/README.md#experiment-reload)。

**例子。** 新 Start 用目前 workflow 附件與固定 capture 轉換 forms，建立 detached native plan、tunables、seed 和 exact run roots，交同一 Controller。Headless owner 持有該 run 的 binding 和 Start capture，不用新 catalog 或目前 draft 重建。附件配對和接線見 [Sources and workflow](../../lib/zcu_tools/gui/app/measure/README.md#sources-and-workflow)。

**邊界。** Caller 提供附件清單，headless facet 驗證名稱唯一。替換清單前必須關閉全部 workflow owners，不保留獨立 workflows app 的 panel reload／持久 input。Experiment reload 不因此取得 workflow 草稿的寫入權。Saved run 由獨立工具讀 manifest 與 iteration records 轉換，GUI／Controller 不保留 export binding，見 [[0062]]。不為這些接線增加 RPC listing、wire schema 或 Registry lookup。

**依據。** 2026-10-09，使用者對共用 source-loading、各 app 自建 catalog 答「A」。這讓 workflows 沿用 loader，Measure 的 recovery 與 UI 留在 app。原裁決未附入庫討論，見 Git `6e9297d2b`；共用 loader 的實作見 `d105f0d1f`。

## 取捨與相鄰文件

不維護 service tier 表，因為它不能解釋單條依賴的用途；也不把所有呼叫塞進 Controller，以免放大依賴並模糊寫入 owner。Frontend 可採不同呈現，不能另存可提交的業務狀態。

Cfg 的編輯和 lowering 見 [[0065]]；持久化見 [[0063]]；workflow 見 [[0062]]；process composition 見 [[0064]]；operation owner-loop、取消與 settle 見 [[0066]]；scheduler 與 gate presence 見 [[0053]]。Remote／Transport 協定見 [[0068]]。本篇不保證各 app 具有同樣的能力，也不保證所有可變繪圖資料都有跨線程 snapshot。
