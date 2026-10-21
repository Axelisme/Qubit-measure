# gui/app/workflows — Workflow host

**Last updated:** 2026-10-22

**讀者：** framework-dev。Workflow GUI 的註冊、表單、執行與繪圖分工。

## 註冊與組裝

`WorkflowAttachment` 配對已宣告的 native step、plan/tunables FormModel、converter 與 context seed。每個附件保留 native plan、tunables、state、record 與 form 的型別配對。同一宿主的附件共享 context 型別參數，宿主不解讀 context，也不依 workflow 名稱分支。註冊項目沒有 export 或 presentation callback。

`WorkflowComposition(registrations=..., templates=..., loader=..., alias_presets=...)` 保存 caller 的 workflow/form 與呈現來源，不建立 live session。`WorkflowsGuiBehavior(composition=...)` 建立一份 EMPTY session。Behavior 的公開 `alias_presets` 欄位保留完整名稱與有序表。

Session 只以 default rows 作未開 entry 的 memory 初值。開 entry 清空表，選 point 載入該 point 的 alias，缺檔空、不回退 preset。Inspect Apply 只保存目前 point，未選 point 回 no_context。省略或 None 建立空表，顯式 mapping 必須含 default，default 的空 tuple 合法。

`WorkflowWindow` 也可接收既有 state/services/owner 與同型別的 `composition`，嵌入 caller 的 Qt parent。清單必須非空且名稱唯一，第一項是預設。GUI、RPC 與 MCP 使用同一 Window 與 Controller。公開參數、欄位、錯誤與 lifetime 契約見各型別 docstring。

`WorkflowsGuiBehavior` 可注入 session 既有的 `DriverFactoryPort`，省略時使用 production driver factory。注入只替換 driver 建立，不改 DeviceService 的新增規則。Behavior 仍擁有 registry，關閉時先 drain workers 再關閉所有已登記 driver。

`scripts/run_workflows_gui.py` 是組合根。它讀 `zcu_lab.workflows.registry`，注入註冊清單、alias preset 與 `SourceCatalogLoader`。它也把 `zcu_lab.templates.TEMPLATES` 傳給 Behavior，與 measure 使用同一份範本目錄。Builder 每次重新 import 當前清單。Framework 和 component sources 固定，workflow/form/registry sources 一起 bounded reload。Framework 不 import 使用者套件，依賴方向由 C16 保護。

## Window 版面

`layout.py` 建一次控制項與版面，`session_strip.py` 的 `WorkflowSessionStrip` 顯示 session 列。最上面一列是 Entry chip（`Entry: <entry> / <point>`）、Setup…、SoC…、Inspect…、Devices…、Prepare forms 與 Reload source。Chip 讀 services，收到 context 切換事件時刷新，Setup 關閉與 Prepare forms 後也刷新。過長文字截斷，tooltip 顯示全文。Setup 管理 entry 與 working points，SoC 顯示連線表單與 soccfg，Devices 顯示裝置狀態。這些按鈕開啟與 Measure 相同的共用對話框。

宿主從 Start 到 terminal delivery 對 Setup、SoC、Inspect、Devices 與 ARB Waveforms 套用 `set_read_only`，終局後恢復編輯。Run 中仍可開啟、瀏覽與關閉這些對話框，提交控制項停用。新開的對話框套用當下的鎖。Prepare forms 與 Reload source 在此期間停用。Reload 另外要求有 loader，且目前沒有要求重啟的 reload 失敗。Window 關閉時一併關閉保留的 Setup 與 SoC 對話框。

下方是可拖動的 QSplitter。左欄依序是 workflow 選單、`Plan`／`Tunables` 兩個分頁與固定在底部的 Start／Pause／Resume／Stop。分頁內容放在 QScrollArea，所以表單展開多少都不影響視窗最小高度；切換 workflow、Prepare 與 Reload 換掉的 panel 放回同一個分頁。右欄依序是 run 狀態列（`run_status.py`）、診斷訊息列 `workflow.run.message` 與 `figure_area.py` 的 `WorkflowFigureArea`。版面只依 widget 結構，不依 workflow 名稱；既有 objectName 保持不變。

`run_view.py` 的 `WorkflowRunView` 把 Controller 目前的 run 呈現在這些控制項上。它讀 Controller 與 forms，不對 run 下指令；它唯一的寫入是 terminal delivery 時把 Start knobs 重設成這次 run 接受的 input。狀態列是徽章（ready、pending、running、pausing、paused、stopping、done、aborted、stopped、failed）、workflow native 名稱、「n committed」與 run_id；宿主不知道總點數。選單選的 workflow 與 attached run 不同時，狀態列前面加 `Previous run:`，圖區仍是那個 run 的圖。輪詢只寫狀態列；Start 的 FormErrors 與例外、指令錯誤、terminal 錯誤或 native failed／aborted 原因寫在下方的診斷列，不被輪詢蓋掉。從 Start 到 terminal delivery，workflow selector 與 Start 停用；Plan 與 Start Tunables 的 editor 呼叫 `set_editing_enabled(enabled=False)` 變成唯讀，面板不停用，所以值不變灰，仍可捲動、展開折疊與選取複製文字，但任何點擊、滾輪或鍵盤都改不了表單；Plan 分頁上方註明 plan 已固定；run 被 accept 後，Tunables 分頁在同一位置改放這個 run 的 tuning panel，分頁名稱是 `Tunables (live)`，Start knobs 隱藏，所以畫面上只有一份 tunables。Running、pausing、paused 時 tuning panel 可編輯，Apply 與 Discard 可用（live tuning）；stopping 時它的 editor 呼叫 `set_editing_enabled(enabled=False)`，Apply 與 Discard 停用，面板不停用，所以值不變灰、仍可折疊與選取。Paused delivery 自動切到這個分頁，並顯示「Resume 會重做未提交的點、可以先 Apply」的提示，送出 Resume 後提示消失。Terminal delivery 換回 Start knobs，內容重設成這次 run 最後接受的 input，可以再編輯並 Start，保存與切換 workflow 時也讀這份 Start knobs；tuning panel 隱藏但保留，讓 RPC 的 form 與 Apply 照舊對到這個 run。Stop 只在 pending、running、pausing、paused 時啟用；Pause 在 running、Resume 在 delivered paused 時啟用。

## Run 與表單

`WorkflowController` 在 Window 的生命週期內保持固定。每次 Start 交入 EnginePorts、pending TunablesHost 與固定 capture。一 run 只有一 operation。Native Engine 的 status、records 與 artifacts 是權威，operation token 只是提交 receipt。Replacement 等待 terminal cleanup。

`TunablesHost` 擁有該 run 唯一的 FormState、固定 evaluation capture 與 accepted expression input。GUI 與 RPC Apply 共用 Controller entry。TunablesHost 以驗證後的完整 Form 呼叫 WorkflowAttachment 的 knobs_to_cfg，與 Start 使用同一個 Form.to_cfg() 宣告。只有 Engine 成功 publication 才接受 input/base；invalid input、轉接失敗與 native failure 保留可修正 draft。Engine 擁有 numeric validation、revision、step capture 與 journal publication。沒有 holder enforcement 或 GUI revision guard。

Pause 保留 operation、RUN lease、progress、plan、input capture 與 Plots。Resume 重用同一 token 和 settlement，不重新 init 或 capture。等待 Ask 時的 Stop 只呼叫 Engine.stop 設 stopping，再以同一 operation 排 Engine.drain worker。作者在 worker 收到 None 並收尾，owner thread 不推進 generator。Controller 保留 token、settlement、RUN lease 與 view，直到 worker 返回且 terminal owner delivery 完成。此時 Close 先請求 Stop，保留視窗供 delivery；之後可再次關閉。Drain 提交失敗保留所有權，Stop／Close 可重試。Terminal delivery 先清理原生 run 與 admission，再通知 view；view 錯誤只記錄，不改 native outcome。SessionDevices 使用 shared public drivers，每次 env.run callback 前讀 connected device evidence。Driver 擁有硬體限制、單位與 setup。Window 在 terminal delivery 後請 DeviceService 背景讀一次已連線儀器，成功台更新觀察 cache，失敗台保留既有值。Pause 不觸發這項讀回。讀回不改寫 acquisition evidence、native outcome 或原本的 settlement／lease 職責。

## Input 與 reload

Wire 5 的 `workflow.form` 回 input／values／errors 與 native revision，不帶 schema。`workflow.describe_form(form, path, run_id, batch)` 沿同一 target selection 和固定 env，在同一 owner turn 回 candidate observation 與單層 description。Root path 是 `[]`，其他 path 使用 str 欄位名與 model-list 的 int index。Read 納入 local parse errors。Batch preview 只讀 parsed candidate，不提交。

Sweep 表單沿用 `gui.form_adapter` 的 Range／Center × Expts／Step input 與 sweep row commands，Qt 只派發 owner 指令。Step 不整除時顯示共用 Form 的有效端點，不自行換算。Memento 保存 active 表示與 expression，Prepare forms 恢復 fresh defaults。Description 的 field commands 保留 target 與參與 inputs；selector 只控制 dropdown 呈現。

Description 的 options 只帶 value／label／commands，selected index 由 core 判定，children 保留完整 path。can_expand 只表示結構可展開，不保證 children 非空。selection_fixed 表示本層只有一個非 null selector 分支，不凍結內層 choices。null 只控制啟停，不算另一個分支，停用時也一樣。fixed／fixed_value 表示完整值固定。Wire 直接投影 core 的這些能力，不重算規則。未知、inactive 或穿入 whole leaf 的 path 以 FormPathError 回報。RPC 將它投影為 invalid_params／form_path，data 帶完整 form_path。其他 ValueError 保留 controller_error。

Model-list 的 description 直接投影 elements，包含原始 index、captured name 與 typed path。GUI／RPC 繼續用 typed paths 編輯共同草稿。Apply 轉換完整 native model，list 隨 model 提交，不攤平 preview leaves、不字串化 index。Engine 驗證完整宣告型別並整體替換，每次成功都增加 revision，同值也一樣。Journal append 成功後才發布值，再接受草稿與 base。當前 step capture 不變。見 ADR-0062、ADR-0079。

Memento 只保留 plan 與 knobs，各欄直接使用固定 FormInput `{data,selections}` JSON。Restore 先依現行 defaults 建立兩個候選，再以 Core 的 `restore_input` 遞迴還原。兩個候選都成功後才重設 input/base。沒有版本或型別 identity，不雙讀舊格式。新欄位用 defaults，移除欄位、形狀不符與非法 selection 列 dropped 並顯示。壞 JSON 或格式不符時，共用 caretaker 顯錯並以 defaults 開啟，不備份。Provider／restore 故障仍傳播。

`WorkflowForms` 用 `WorkflowDraft` 保留各名稱的 parsed RAM FormInputs。Switch/reload 與啟動共用 Core 還原，不呼叫 converter 或 options hook。兩個 candidate panels 建好後才發布。Prepare 用新 capture 重建 defaults。Caretaker 不保存 renderer-local 未解析文字、native run 或硬體。從 Start 到 terminal delivery，保存的 knobs 來自 run binding，包括尚未 Apply 的 parsed 編輯；terminal 之後來自 Start knobs。`--clean` 只略過啟動讀取，正常 close 仍保存。啟動 session 是 EMPTY，還原不查 library。Ref 名稱保留，缺 library 或來源錯誤由一般驗證呈現並擋 Run。已 inline 的原稿不從改過的 library 重建。

Reload 只允許 idle/terminal，保留 session/services/controller、run binding 與既有 captures。相容 input 保留，無效 selection 改用現行 defaults 並列入 dropped。Import/panel 失敗保留舊 panels/drafts，但阻擋 Start 直到 retry 成功；fixed-source 改動要求 restart。

## Plots 與 remote

Workflow 透過 `env.plots` 決定畫什麼與何時更新。Window 只管理 Plots lifetime，不輪詢 committed records 或呼叫科學 trend renderer。Pause 不 finish；producer 返回與 terminal delivery 後 finish，新 Start 或 close release。

`WorkflowFigureArea` 是每個 run 的 Plots 的 PlotHost，包住共用的 `QtPlotHost` 與 `FigureContainer`，所以 measure 用的 `gui/plotting` 不變。Focus 的圖放大，標題是 figure 名稱加「k / n」。下方縮圖列依該 run 的 `env.plots` 建立順序列出已呈現的圖；focus 那張有醒目的外框，列可橫向捲動並停在 focus 那張。Follow latest 預設勾選，focus 跟著剛呈現或剛 live refresh 的圖；run 結束時的 final flush 不算更新。點縮圖固定 focus 並取消勾選，重新勾選回到最近更新的圖，每次 Start 重設為勾選。縮圖從 canvas 重抓的頻率低於大圖，大圖沿用共用 plotting 的更新收合。宿主只依 figure 名稱排列，不依 workflow 名稱分支。

Records RPC 直接投影 Engine facts。具名圖與 PNG 使用同一 retained Plots，在 Qt owner thread render；status-only 不 render。MCP tools 在 `zcu_tools.mcp.workflows`，只轉送同一 wire declarations，不持有第二份 run、draft 或硬體。Agent 組合根在 `mcp_servers/workflows.py`，它注入 bridge 的 repo root 與 GUI 啟動設定，再由 GUI launcher 注入註冊清單與 source loader。新增的單層查詢也由同一 method declarations 產生 MCP tool，不另讀 core 或保存 candidate。Server identity 保持不變。

框架測試在 `tests/gui/app/workflows/` 與 `tests/mcp/workflows/` 使用最小合成註冊清單，不依賴 lab 實驗。Workflow 行為比對在 GUI 的模擬模式完成；宿主不擁有科學 trend 資料或更新時機。

Saved run conversion 由使用者 workflow 的獨立工具負責。生命週期、保存與前端分工見 ADR-0062、ADR-0063、ADR-0067、ADR-0068。
