# GUI applications

**Last updated:** 2026-10-28

**讀者：** framework-dev、agent。查找各 GUI app 的 owner，以及 app 與共用 session 的接縫。

## App owners

| 目錄 | 負責的操作 |
| --- | --- |
| [measure](measure/README.md) | 建立實驗與 workflow tabs，執行量測、分析、保存與寫回。 |
| [fluxdep](fluxdep/README.md) | 載入 fluxonium 光譜，執行選點、擬合與 predictor 匯出。 |
| [dispersive](dispersive/README.md) | 讀取光譜與 predictor，擬合並匯出 dispersive 參數。 |
| [analysis_project](analysis_project/README.md) | 為 fluxdep 與 dispersive 開啟既有 ResultEntry，選取 fluxonium component，提供共用來源選擇。 |

各 app 的 Controller 接收 UI 與 remote 命令，將操作交給同一組 application owners。
State 保存已提交的 app 狀態，UI 與 remote 各自投影結果。
具體實驗與 lab 定義由 repo 根目錄的 `apps/` 組合根注入，框架不反向 import `zcu_lab`。

## 共用接縫

[session](../session/README.md) 擁有 context、SoC、device 與 operation 機制。
Measure 組裝這些能力，shared dialogs 只取得所需的 control facets，不取得整個 app Controller。
Fluxdep 與 dispersive 透過 analysis_project 開啟分析目標，不因此取得 Measure 的量測 session。
共用 session、remote 與 widgets 不依賴 analysis_project。

表單編輯由 [form](../form/README.md) 與 [form_adapter](../form_adapter/README.md) 擁有。
共用圖形呈現由 [plotting](../plotting/README.md) 擁有，app 不另建數值分析核心。

Owner 分工見 [ADR-0067](../../../../docs/adr/0067-gui-application.md)。
套件定位與依賴方向見 [ADR-0072](../../../../docs/adr/0072-module-boundaries.md)。
