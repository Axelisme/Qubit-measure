# gui/app/workflows — Workflow host

**Last updated:** 2026-10-13，stopping 時 live tuning 唯讀但可讀、停用 Apply 與 Discard；run 進行中 Plan 與 Start Tunables 表單唯讀但可讀（不再停用面板）；run 進行中停用 Prepare forms 與 Reload source；terminal 後保存可編輯的 Start knobs；run view 在 terminal 時重設 Start knobs

## 註冊與組裝

`WorkflowAttachment` 配對已宣告的 native step、plan/tunables FormModel、converter 與 context seed。每個附件保留 native plan、tunables、state、record 與 form 的型別配對。同一宿主的附件共享 context 型別參數，宿主不解讀 context，也不依 workflow 名稱分支。註冊項目沒有 export 或 presentation callback。

`WorkflowsGuiBehavior(registrations=..., loader=..., alias_rows=...)` 建立一份 EMPTY session。`WorkflowWindow` 也可接收既有 state/services/owner 與註冊清單，嵌入 caller 的 Qt parent。清單必須非空且名稱唯一，第一項是預設。GUI、RPC 與 MCP 使用同一 Window 與 Controller。公開參數、欄位、錯誤與 lifetime 契約見各型別 docstring。

`scripts/run_workflows_gui.py` 是組合根。它讀 `zcu_lab.workflows.registry`，注入註冊清單、alias preset 與 `SourceCatalogLoader`。Builder 每次重新 import 當前清單。Framework 和 component sources 固定，workflow/form/registry sources 一起 bounded reload。Framework 不 import 使用者套件，依賴方向由 C16 保護。

## Window 版面

`layout.py` 建一次控制項與版面，`session_strip.py` 的 `WorkflowSessionStrip` 顯示 session chips。最上面一列是 chips（Entry／Point、`$qubit` 解析出的元件與 kind、已連線裝置、ZCU 狀態）與 Setup…、Aliases…、Prepare forms、Reload source。Chips 只讀 services，收到 context、alias、point 內容、SoC、裝置事件時刷新，Setup／Aliases 關閉與 Prepare forms 後也刷新；過長的診斷截斷，完整文字放在 tooltip，不撐寬視窗。從 Start 到 terminal delivery（pending、running、pausing、paused、stopping），宿主拒絕 Prepare forms 與 Reload source，所以這兩顆停用；Setup… 與 Aliases… 改的是 session，不影響已 capture 的 run，維持可按。Reload 另外要有 loader，且沒有要求重啟的 reload 失敗。

下方是可拖動的 QSplitter。左欄依序是 workflow 選單、`Plan`／`Tunables` 兩個分頁與固定在底部的 Start／Pause／Resume／Stop。分頁內容放在 QScrollArea，所以表單展開多少都不影響視窗最小高度；切換 workflow、Prepare 與 Reload 換掉的 panel 放回同一個分頁。右欄依序是 run 狀態列（`run_status.py`）、診斷訊息列 `workflow.run.message` 與 `figure_area.py` 的 `WorkflowFigureArea`。版面只依 widget 結構，不依 workflow 名稱；既有 objectName 保持不變。

`run_view.py` 的 `WorkflowRunView` 把 Controller 目前的 run 呈現在這些控制項上。它讀 Controller 與 forms，不對 run 下指令；它唯一的寫入是 terminal delivery 時把 Start knobs 重設成這次 run 接受的 input。狀態列是徽章（ready、pending、running、pausing、paused、stopping、done、aborted、stopped、failed）、workflow native 名稱、「n committed」與 run_id；宿主不知道總點數。選單選的 workflow 與 attached run 不同時，狀態列前面加 `Previous run:`，圖區仍是那個 run 的圖。輪詢只寫狀態列；Start 的 FormErrors 與例外、指令錯誤、terminal 錯誤或 native failed／aborted 原因寫在下方的診斷列，不被輪詢蓋掉。從 Start 到 terminal delivery，workflow selector 與 Start 停用；Plan 與 Start Tunables 的 editor 呼叫 `set_editing_enabled(enabled=False)` 變成唯讀，面板不停用，所以值不變灰，仍可捲動、展開折疊與選取複製文字，但任何點擊、滾輪或鍵盤都改不了表單；Plan 分頁上方註明 plan 已固定；run 被 accept 後，Tunables 分頁在同一位置改放這個 run 的 tuning panel，分頁名稱是 `Tunables (live)`，Start knobs 隱藏，所以畫面上只有一份 tunables。Running、pausing、paused 時 tuning panel 可編輯，Apply 與 Discard 可用（live tuning）；stopping 時它的 editor 呼叫 `set_editing_enabled(enabled=False)`，Apply 與 Discard 停用，面板不停用，所以值不變灰、仍可折疊與選取。Paused delivery 自動切到這個分頁，並顯示「Resume 會重做未提交的點、可以先 Apply」的提示，送出 Resume 後提示消失。Terminal delivery 換回 Start knobs，內容重設成這次 run 最後接受的 input，可以再編輯並 Start，保存與切換 workflow 時也讀這份 Start knobs；tuning panel 隱藏但保留，讓 RPC 的 form 與 Apply 照舊對到這個 run。Stop 只在 pending、running、pausing、paused 時啟用；Pause 在 running、Resume 在 delivered paused 時啟用。

## Run 與表單

`WorkflowController` 在 Window 的生命週期內保持固定。每次 Start 交入 EnginePorts、pending TunablesHost 與固定 capture。一 run 只有一 operation。Native Engine 的 status、records 與 artifacts 是權威，operation token 只是提交 receipt。Replacement 等待 terminal cleanup。

`TunablesHost` 擁有該 run 唯一的 FormState、固定 evaluation capture 與 accepted expression input。GUI 與 RPC Apply 共用 Controller entry。只有 Engine 成功 publication 才接受 input/base；invalid input 保留 draft。Engine 擁有 numeric validation、revision、step capture 與 journal publication。沒有 holder enforcement 或 GUI revision guard。

Pause 保留 operation、RUN lease、progress、plan、input capture 與 Plots。Resume 重用同一 token 和 settlement，不重新 init 或 capture。Terminal delivery 先清理原生 run 與 admission，再通知 view；view 錯誤只記錄，不改 native outcome。SessionDevices 使用 shared public drivers，每次 env.run callback 前讀 connected device evidence。Driver 擁有硬體限制、單位與 setup。

## Input 與 reload

`WorkflowForms` 保留各名稱的 parsed RAM drafts，建好候選 panels 後才發布。Prepare 用新 capture 重建 defaults。`SingleFileCaretaker` 保存 parsed FormInput，包括 expression，不保存 renderer-local 未解析文字、native run 或硬體。從 Start 到 terminal delivery，保存的 knobs 來自 run 的 binding，包括還沒 Apply 的 parsed 編輯；terminal 之後來自 Start knobs。`--clean` 只略過啟動讀取，正常 close 仍保存。啟動時 session 是 EMPTY、沒有 captured library，所以存檔裡的 module ref 名稱會被丟棄，其他相容欄位照常還原，訊息列逐一顯示 `Discarded plan: …` 或 `Discarded knobs: …`。

Reload 只允許 idle/terminal，保留 session/services/controller、run binding 與既有 captures。相容 input 保留，removed selection 改選清單第一項。Import/panel 失敗保留舊 panels/drafts，但阻擋 Start 直到 retry 成功；fixed-source 改動要求 restart。

## Plots 與 remote

Workflow 透過 `env.plots` 決定畫什麼與何時更新。Window 只管理 Plots lifetime，不輪詢 committed records 或呼叫科學 trend renderer。Pause 不 finish；producer 返回與 terminal delivery 後 finish，新 Start 或 close release。

`WorkflowFigureArea` 是每個 run 的 Plots 的 PlotHost，包住共用的 `QtPlotHost` 與 `FigureContainer`，所以 measure 用的 `gui/plotting` 不變。Focus 的圖放大，標題是 figure 名稱加「k / n」。下方縮圖列依該 run 的 `env.plots` 建立順序列出已呈現的圖；focus 那張有醒目的外框，列可橫向捲動並停在 focus 那張。Follow latest 預設勾選，focus 跟著剛呈現或剛 live refresh 的圖；run 結束時的 final flush 不算更新。點縮圖固定 focus 並取消勾選，重新勾選回到最近更新的圖，每次 Start 重設為勾選。縮圖從 canvas 重抓的頻率低於大圖，大圖沿用共用 plotting 的更新收合。宿主只依 figure 名稱排列，不依 workflow 名稱分支。

Records RPC 直接投影 Engine facts。具名圖與 PNG 使用同一 retained Plots，在 Qt owner thread render；status-only 不 render。MCP tools 在 `zcu_tools.mcp.workflows`，只轉送同一 wire declarations，不持有第二份 run、draft 或硬體。Agent 組合根在 `mcp_servers/workflows.py`，它注入 bridge 的 repo root 與 GUI 啟動設定，再由 GUI launcher 注入註冊清單與 source loader。RPC 方法、參數、回覆與 server identity 保持不變。

框架測試在 `tests/gui/app/workflows/` 與 `tests/mcp/workflows/` 使用最小合成註冊清單，不依賴 lab 實驗。Workflow 行為比對在 GUI 的模擬模式完成；宿主不擁有科學 trend 資料或更新時機。

Saved run conversion 由使用者 workflow 的獨立工具負責，GUI/RPC/MCP 不提供 export commands。生命週期、保存與前端分工見 ADR-0062、ADR-0063、ADR-0067、ADR-0068。
