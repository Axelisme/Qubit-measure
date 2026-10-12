# gui/app/workflows — Workflow host

**Last updated:** 2026-10-13，圖區改成 focus 大圖加縮圖列

## 註冊與組裝

`WorkflowAttachment` 配對已宣告的 native step、plan/tunables FormModel、converter 與 context seed。每個附件保留 native plan、tunables、state、record 與 form 的型別配對。同一宿主的附件共享 context 型別參數，宿主不解讀 context，也不依 workflow 名稱分支。註冊項目沒有 export 或 presentation callback。

`WorkflowsGuiBehavior(registrations=..., loader=..., alias_rows=...)` 建立一份 EMPTY session。`WorkflowWindow` 也可接收既有 state/services/owner 與註冊清單，嵌入 caller 的 Qt parent。清單必須非空且名稱唯一，第一項是預設。GUI、RPC 與 MCP 使用同一 Window 與 Controller。公開參數、欄位、錯誤與 lifetime 契約見各型別 docstring。

`scripts/run_workflows_gui.py` 是組合根。它讀 `zcu_lab.workflows.registry`，注入註冊清單、alias preset 與 `SourceCatalogLoader`。Builder 每次重新 import 當前清單。Framework 和 component sources 固定，workflow/form/registry sources 一起 bounded reload。Framework 不 import 使用者套件，依賴方向由 C16 保護。

## Window 版面

`layout.py` 建一次控制項與版面，`session_strip.py` 的 `WorkflowSessionStrip` 顯示 session chips。最上面一列是 chips（Entry／Point、`$qubit` 解析出的元件與 kind、已連線裝置、ZCU 狀態）與 Setup…、Aliases…、Prepare forms、Reload source。Chips 只讀 services，收到 context、alias、point 內容、SoC、裝置事件時刷新，Setup／Aliases 關閉與 Prepare forms 後也刷新；過長的診斷截斷，完整文字放在 tooltip，不撐寬視窗。

下方是可拖動的 QSplitter。左欄依序是 workflow 選單、`Plan`／`Tunables` 兩個分頁與固定在底部的 Start／Pause／Resume／Stop。分頁內容放在 QScrollArea，所以表單展開多少都不影響視窗最小高度；切換 workflow、Prepare 與 Reload 換掉的 panel 放回同一個分頁。Accepted run 的 tuning panel 放在 Tunables 分頁、Start knobs 的下方。右欄是 run 訊息列與 `figure_area.py` 的 `WorkflowFigureArea`。版面只依 widget 結構，不依 workflow 名稱；既有 objectName 保持不變。

## Run 與表單

`WorkflowController` 在 Window 的生命週期內保持固定。每次 Start 交入 EnginePorts、pending TunablesHost 與固定 capture。一 run 只有一 operation。Native Engine 的 status、records 與 artifacts 是權威，operation token 只是提交 receipt。Replacement 等待 terminal cleanup。

`TunablesHost` 擁有該 run 唯一的 FormState、固定 evaluation capture 與 accepted expression input。GUI 與 RPC Apply 共用 Controller entry。只有 Engine 成功 publication 才接受 input/base；invalid input 保留 draft。Engine 擁有 numeric validation、revision、step capture 與 journal publication。沒有 holder enforcement 或 GUI revision guard。

Pause 保留 operation、RUN lease、progress、plan、input capture 與 Plots。Resume 重用同一 token 和 settlement，不重新 init 或 capture。Terminal delivery 先清理原生 run 與 admission，再通知 view；view 錯誤只記錄，不改 native outcome。SessionDevices 使用 shared public drivers，每次 env.run callback 前讀 connected device evidence。Driver 擁有硬體限制、單位與 setup。

## Input 與 reload

`WorkflowForms` 保留各名稱的 parsed RAM drafts，建好候選 panels 後才發布。Prepare 用新 capture 重建 defaults。`SingleFileCaretaker` 保存 parsed FormInput，包括 expression，不保存 renderer-local 未解析文字、native run 或硬體。`--clean` 只略過啟動讀取，正常 close 仍保存。啟動時 session 是 EMPTY、沒有 captured library，所以存檔裡的 module ref 名稱會被丟棄，其他相容欄位照常還原，訊息列逐一顯示 `Discarded plan: …` 或 `Discarded knobs: …`。

Reload 只允許 idle/terminal，保留 session/services/controller、run binding 與既有 captures。相容 input 保留，removed selection 改選清單第一項。Import/panel 失敗保留舊 panels/drafts，但阻擋 Start 直到 retry 成功；fixed-source 改動要求 restart。

## Plots 與 remote

Workflow 透過 `env.plots` 決定畫什麼與何時更新。Window 只管理 Plots lifetime，不輪詢 committed records 或呼叫科學 trend renderer。Pause 不 finish；producer 返回與 terminal delivery 後 finish，新 Start 或 close release。

`WorkflowFigureArea` 是每個 run 的 Plots 的 PlotHost，包住共用的 `QtPlotHost` 與 `FigureContainer`，所以 measure 用的 `gui/plotting` 不變。Focus 的圖放大，標題是 figure 名稱加「k / n」。下方縮圖列依該 run 的 `env.plots` 建立順序列出已呈現的圖；focus 那張有藍框，列可橫向捲動並停在 focus 那張。Follow latest 預設勾選，focus 跟著剛呈現或剛 live refresh 的圖；run 結束時的 final flush 不算更新。點縮圖固定 focus 並取消勾選，重新勾選回到最近更新的圖，每次 Start 重設為勾選。縮圖每 250 ms 至多從 canvas 重抓一次，大圖沿用共用的 50 ms 收合。宿主只依 figure 名稱排列，不依 workflow 名稱分支。

Records RPC 直接投影 Engine facts。具名圖與 PNG 使用同一 retained Plots，在 Qt owner thread render；status-only 不 render。MCP tools 在 `zcu_tools.mcp.workflows`，只轉送同一 wire declarations，不持有第二份 run、draft 或硬體。RPC 方法、參數、回覆與 server identity 保持不變。

Saved run conversion 由使用者 workflow 的獨立工具負責，GUI/RPC/MCP 不提供 export commands。生命週期、保存與前端分工見 ADR-0062、ADR-0063、ADR-0067、ADR-0068。
