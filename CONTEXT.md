# 專案詞彙

本表定義 zcu_tools 與 zcu_lab 共用的詞彙，連結指向各概念的 owner 文件。

## Owner

對某份狀態、資源或行為具有寫入與生命週期決定權的模組或服務。
其他模組透過 owner 的命令、查詢或已提交的 domain fact 合作，不直接接管其狀態，見 [ADR-0067](docs/adr/0067-gui-application.md)。

## Adapter

實驗核心與前端之間的接合層，GUI 與 Notebook 各自包裝核心，不互相包裝，見 [ADR-0069](docs/adr/0069-experiment-core-and-frontends.md)。
GUI adapter 依框架契約宣告能力並提供執行、分析與保存內容，能力宣告不等於當次 readiness 或硬體權限，見 [ADR-0067](docs/adr/0067-gui-application.md)。

## Handle

Operation handle 是追蹤工作終局、等待與取消的識別，不承載完整實驗結果，也不是持有硬體 lease 的證明，見 [ADR-0066](docs/adr/0066-operation-lifecycle.md)。
Run／Load result 的 opaque handle 則由 adapter 擁有內容，GUI 框架不探測其 cfg 或 result 欄位，見 [ADR-0067](docs/adr/0067-gui-application.md)。

## Workflow

由宿主執行的具名多步實驗流程，可在步驟間保留已提交狀態並 Pause／Resume，見 [workflow 核心](lib/zcu_tools/experiment/workflows/README.md)。
具體 workflow 擁有科學流程與量測產物的保存決策，宿主擁有硬體連線與執行資源，見 [ADR-0062](docs/adr/0062-experiment-workflow.md)。

## Writeback

把分析成果以 Parameter、Module 或 Sample proposal 提交回固定來源的流程，不是保存量測資料檔。
選定 Apply 依序提交提案，失敗時保留成功前綴，不提供跨項 rollback，見 [ADR-0008](docs/adr/0008-cfg-editor-session.md) 與 [ADR-0067](docs/adr/0067-gui-application.md)。

## Recipe

透過既有 Measure GUI 操作串成的有界實驗流程，參數來源、缺參數與分析分支由個別 recipe 宣告，見 [Measure MCP](lib/zcu_tools/mcp/measure/README.md)。
具體流程由 [zcu_lab](zcu_lab/README.md) 擁有，此處的 recipe 不指 waveform 資產中的 formula recipe。

## Component

參數文件中具名、帶 kind 的實驗領域對象，例如 qubit、resonator 或 mux group。
[Resources entry](lib/zcu_tools/resources/entry/README.md) 提供容器與註冊契約，具體欄位、關聯與 module 槽由 [zcu_lab](zcu_lab/README.md) 的元件定義擁有。

## Module

用於 pulse、readout 或 reset 的可重用設定單位，不是 Python module 或 package。
持久 ModuleSpec 保留具體 key 的 child 引用，ModuleCfg 則是已展開的執行參數，見 [cfg_model](lib/zcu_tools/cfg_model/README.md) 與 [point-local library](lib/zcu_tools/resources/context/README.md)。

## Alias

GUI 全域有序表中的具名綁定，值是 component 或 device 名稱、未綁定的空字串，或引用前列的 `$` 軟連結。
ContextService 擁有此表，Run／Load 後的分析、Writeback 與 Save 使用固定來源，不隨後來的表格編輯改變，見 [ADR-0067](docs/adr/0067-gui-application.md)。

## RunRecord

配對單次量測的 nullable typed cfg 與實驗專屬 Result 的資料紀錄，不包含硬體 handle 或呈現資源。
載入資料時 cfg 不相容可為 None，量測資料仍可使用，見 [experiment](lib/zcu_tools/experiment/README.md) 與 [ADR-0069](docs/adr/0069-experiment-core-and-frontends.md)。

## Artifact

一次寫入的單位，內部可包含 DataItem、FigureItem 與 AnalysisItem，不等於單一檔案。
Adapter 只交內容，框架決定格式、路徑與逐項保存，見 [ADR-0067](docs/adr/0067-gui-application.md) 與 [experiment](lib/zcu_tools/experiment/README.md)。
