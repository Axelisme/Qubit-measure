# GUI adapter capability 入口檢查

**狀態：** 已核准待實作；以下不是現行 application guard 保證。現行各入口的檢查見 [ADR-0067](../0067-gui-application.md)。

## 分析與後分析的 application 入口

核准目標：GUI、remote 與 application 操作入口使用 adapter 宣告的 analysis／post-analysis capability 拒絕不支援的操作；UI 控制項的存在與否不能充當唯一守門。Capability 與當次 context readiness、結果可用性、busy 及硬體互斥分開判斷。不靠 method presence 推測支援範圍。

現況證據：`gui/app/measure/ui/exp_tab_widget.py` 依 `AdapterCapabilities.analysis`／`post_analysis` 建立控制項；`gui/app/measure/remote/handlers/writeback.py` 只對 writeback subtab params 檢查對應 capability。`gui/app/measure/services/guard.py::acquire_analyze_permit` 只查 tab、context 和 run result，未查 analysis capability。`gui/app/measure/services/run_analyze_control.py::analyze` 只把 `INTERACTIVE` 分流，其餘（包括 `NONE`）走 FIT worker；`start_post_analyze` 未查 post-analysis capability。`gui/app/measure/services/post_analyze.py::start_post_analyze` 只查 busy 和 primary analyze result。Run guard 已按 `requires_soc` 檢查連線；load permit 和 `LoadService` 已按 `load_data` 拒絕不支援的 load，不把已實作的局部檢查說成沒有 guard。

**轉正條件：** 核實 GUI、remote 及直接 application 呼叫對不支援 analysis／post-analysis 的一致拒絕，且不會把 `AnalysisMode.NONE` 送到 FIT worker；保留既有 run／load 與 readiness、結果和動態 busy 的區分。以入口行為核實後，才把完整 capability guard 寫回現行 ADR 與 owner 文件。
