# Measure services

**Last updated:** 2026-11-01

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
Ledger 讀取使用 resources 的無 lock typed partial snapshot，只消費 acquired 身分／時間與 saved 成功路徑。未用的歷史 payload 或 attachment reference 不影響登錄；實際使用的壞值仍診斷並沿既有 entry 層級隔離。
沒有對應 Result entry 的 Database run 仍可瀏覽。無法安全列舉的 Database 項目
以 `ResultsIndex.issues` 回報 entry、路徑與原因，不偽造 run；可列出的 run
則沿用 `ResultRun.issues`。Symlink escape 項目不讀、不跟隨。

`RunService` 在提交 worker 前固定 ParameterContext 的參數快照。Worker 以 RunMetadata 組內容並將 JSON dump 放入 record.metadata，後續切換 point 不改這份分析環境。
RunService 自己記錄 Stop 與成功的 selector cancel，交付 stop_requested。
Core 正常回傳就發布並標 finished，不區分 root 或子孫停止。拋錯就 failed 並顯示錯誤。
Stop 與被接受的 selector cancel 以 stop_requested 記錄。
Core 回傳前接受的請求使存檔 completion 為 stopped。
它不讀 StopSignal 的完成或錯誤事實。

Load 只使用檔內 experiment tag、資料與參數快照，不讀 live point 或先驗完整 RunMetadata。缺快照使用空參數內容；壞快照由 owner 報錯，Load 保留原 pane 並回傳指名來源檔的 LoadDataError。未用的歷史 metadata 欄位延後到使用或完整投影時驗證。Analyze 與 interactive analysis 固定 detached metadata，將裝置證據 reader 交給 draft。Post 延續 Primary 的 reader，不在啟動時投影裝置設定。真正讀取 draft.device_snapshot 時仍由 experiment owner 驗證並回報欄位路徑。

Run、Analyze、PostAnalyze、Load 與 Tab 將 pane 提交或退役清理交給 `pane_commit`。
Composition root 將同一個 `SaveService.plan_layout` 注入四個結果 producers，預覽與 Save 共用 adapter stem 及 `bind_run_outputs` 的路徑計算。
正常 Load 後立即顯示預覽，Analyze／PostAnalyze 後也顯示，不要求 storage 或 live entry／point。
預覽只使用計畫的 layout，不建立目錄或改變未保存狀態。
Metadata 缺 native schema 必填 key 時沒有預覽，也不追加 acquired。逐欄讀取的 run_id、experiment 與 device_snapshot 不受其他缺 key 影響。
歷史命名欄位讀取失敗時，預覽只隔離 `MetadataFieldError`，保留 Load 與原始內容；其他規劃錯誤照常傳出。
Save 仍完整驗證 metadata，report 發布實際輸出與逐項失敗。
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
