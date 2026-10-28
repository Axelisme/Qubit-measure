# Measure services

**Last updated:** 2026-10-28

**讀者：** framework-dev、agent。查找 Measure application 操作的 owner，以及 service 與 session、UI、remote 的接縫。

## 操作 owners

`app_services.py` 建立 `AppServices`，將同一組 owners 交給 Controller。
UI 與 remote 經 Controller／control facets 呼叫這些 owners，不各自建立量測流程。

| 模組 | 負責的操作 |
| --- | --- |
| `tab.py`、`tab_form.py`、`tab_input.py` | 建立與關閉實驗 tab，捕捉表單來源並管理輸入。 |
| `run.py`、`analyze.py`、`post_analyze.py` | 執行 Run、primary analysis 與 post-analysis，透過 State 提交 pane 成果。 |
| `load.py` | 將已保存的成果交回 app 的 tab／pane 流程。 |
| `workflow_control.py`、`workflow_run.py`、`workflow_tab.py` | 將 workflow 註冊、執行與 tab 交接接到同一份 session。 |
| `startup.py`、`experiment_reload.py`、`workspace.py` | 執行啟動、重載與目前程序內的 workspace 重建。 |
| `results_index.py`、`results_comparison.py` | 唯讀索引保存成果，將已存 summary 投影為比較列，不重新分析。 |

State 擁有已提交的 tab 與 pane 狀態。
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
