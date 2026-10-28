# Measure services

**Last updated:** 2026-10-29

**讀者：** framework-dev、agent。查找 Measure application 操作的 owner，以及 service 與 session、UI、remote 的接縫。

## 操作 owners

`app_services.py` 建立 `AppServices`，將同一組 owners 交給 Controller。
UI 與 remote 經 Controller／control facets 呼叫這些 owners，不各自建立量測流程。

| 模組 | 負責的操作 |
| --- | --- |
| `tab.py`、`tab_form.py`、`tab_input.py` | 建立與關閉實驗 tab，捕捉表單來源並管理輸入。 |
| `guard.py` | 統一 tab action 的 domain 前提與 tab busy 判斷，回傳 admission decision。 |
| `run.py`、`analyze.py`、`post_analyze.py` | 執行 Run、primary analysis 與 post-analysis。 |
| [`pane_commit.py`](pane_commit.py) | 規劃 artifacts、提交 pane、遞增版本並清理退役資源。 |
| `load.py` | 將已保存的成果交回 app 的 tab／pane 流程。 |
| `workflow_control.py`、`workflow_run.py`、`workflow_tab.py` | 將 workflow 註冊、執行與 tab 交接接到同一份 session。 |
| `startup.py`、`experiment_reload.py`、`workspace.py` | 執行啟動、重載與目前程序內的 workspace 重建。 |
| `results_index.py`、`results_comparison.py` | 唯讀索引保存成果，將已存 summary 投影為比較列，不重新分析。 |

Results index 隔離各 entry 的 setup／ledger 與 run 成員讀取錯誤，保留其他成果。
沒有對應 Result entry 的 Database run 仍可瀏覽。無法安全列舉的 Database 項目
以 `ResultsIndex.issues` 回報 entry、路徑與原因，不偽造 run；可列出的 run
則沿用 `ResultRun.issues`。Symlink escape 項目不讀、不跟隨。

`RunService` 在提交 worker 前固定 ParameterContext 的參數快照。Worker 將它放進當次 RunMetadata，後續切換 point 不改這份分析環境。

Load 只使用檔內 experiment tag、資料與參數快照，不讀 live point 或先驗完整 RunMetadata。缺快照使用空參數內容；壞快照由 owner 報錯，Load 保留原 pane 並回傳指名來源檔的 LoadDataError。未用的歷史 metadata 欄位延後到使用或完整投影時驗證。

Run、Analyze、PostAnalyze、Load 與 Tab 將 pane 提交或退役清理交給 `pane_commit`。
`GuardService.admit` 統一 domain admission，啟動時的 hardware exclusion 仍由 operation gate 裁決。
各 operation 決定終局與觸發清理的時機。
State 保存已提交的 tab 與 pane 狀態，只被動交換已備妥的 pane 與 artifacts，回傳退役資源。
State 不規劃 artifacts、不遞增提交版本，也不清理退役資源。
`ports.py` 定義 service 合作時使用的 snapshots 與窄介面，數值分析仍由 adapter 背後的實驗核心負責。

## 共用資源與相鄰 owners

[session](../../../session/README.md) 擁有 context、SoC、device、progress 與 OperationRunner。
Measure 的實驗與 workflow 使用同一份 SessionServices，借用相同 runner、progress 與硬體 lease。
各操作提供自己的 policy，OperationRunner 執行共用的 operation 生命週期。

保存由 [artifacts](../artifacts/) 擁有，proposal／draft／Apply 由 [writeback](../writeback/README.md) 擁有。
App composition 注入它們，不把保存或寫回權威移到 UI。
共用 owner 與 operation 分界見 [Measure README](../README.md#owners)、
[ADR-0067](../../../../../../docs/adr/0067-gui-application.md) 與
[ADR-0066](../../../../../../docs/adr/0066-operation-lifecycle.md)。
