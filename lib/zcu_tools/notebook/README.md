# `zcu_tools.notebook`

**Last updated:** 2026-10-01 — 共用 records 與 typed NotebookAdapter

`zcu_tools.notebook` 提供 Notebook 逐步探索時使用的互動入口、顯示與 widgets，也保留工作流程專用的分析支援。Notebook 工作流程可組合計算與人工確認，不等於 GUI 的量測 session 或狀態管理。實際操作與結果解讀見 [Notebook 內容入口](../../../notebook_md/README.md)；這裡說明支援程式的位置。

## 工作家族

- [`adapter.py`](adapter.py)：`NotebookAdapter` 綁定 experiment instance、可選 hardware handles 與 host。`run()` 需要 soc／soccfg，每次新建 QickContext 與 Plots；`load()`／同步 `analyze()` 可離線使用。RunRecord 組合 nullable cfg 與純資料，AnalysisRecord 組合 explicit source、typed options／analysis 與純具名 figures。同步分析入口只適用於提供 analyze 的核心，不替互動核心補空方法。`save(source, destination)` 不依目前 run，提供 unique path 並回傳實際目的地；canonical cfg 要求仍由核心決定。
- [fluxdep](analysis/fluxdep/README.md)：通量依賴光譜的資料處理、擬合、圖表與互動選點。`fitting.py` 保留 `fit_spectrum` 與組合搜尋、診斷圖的 `search_in_database` 入口；數值搜尋由 analysis 擁有。
- [t1_curve](analysis/t1_curve/README.md) 與 [t2_curve](analysis/t2_curve/README.md)：各自保留 Notebook 的曲線分析、擬合與分階段工作流程。模型選擇與保存條件見各自的 README。
- [fit_tools](analysis/fit_tools/README.md)：支援 Notebook 分析中的校正、資料接合、loss、weights 與溫度模型；不把這些能力一概視為共用分析核心。
- [design](analysis/design/README.md)：評估模型參數、設計需求與候選組合，並分析 HFSS sweep 資料。
- [mist](analysis/mist/tool.py)：提供能量摺疊、不連續處理與碰撞遮罩的計算工具。
- [circuit_design](circuit_design/README.md)：提供 Qiskit Metal 電路幾何元件；相關 Notebook 展示電路建構及設計檔輸出。
一般 T1 使用 `NotebookAdapter(T1Exp())` 與核心的 typed config／options，不另提供扁平參數 wrapper。T1Result 只有資料；RunRecord 是保存與分析的 explicit 來源。Canonical load 可保留 cfg=None 的有效資料，T1 可離線分析，預設 saver 則拒絕缺 cfg 的來源。
- [`experiments/ge.py`](experiments/ge.py)：`GEExp` 使用相同的 Notebook host 呼叫 GE 共用核心，尚未接上新版共用 records。FIT 與 post 各自保留來源、選項、數值與具名圖；post 採用上一次成功的 FIT 校準。失敗操作不覆蓋成功紀錄，run／load 成功清空分析引用，舊圖仍可保存。
- [`experiments/flux_dep.py`](experiments/flux_dep.py)：尚未接上新版共用 records。`FluxDepNotebookExp.analyze()` 回傳選線 widget 與可拖曳的預覽。使用者按 Done 才呼叫核心，並發布 source、options、數值 result 和具名 `pick` Figure。Cancel 或失敗保留舊紀錄。run/load 成功清空目前分析，但使用者仍可保存舊 Figure。預覽 Figure 與 Result 分開。
- [`utils.py`](utils.py)：提供 sweep、圖檔保存與設備資訊等 Notebook 輔助函式。
- [`plotting.py`](plotting.py)：`NotebookPlotHost` 實作共用 `PlotHost`，直接以 ipympl widget 呈現原生 Figure。不登記 pyplot manager，也不切換全域 backend。普通圖與 liveplot 的呈現時機由 `Plots` 控制，host 不偵測 browser 是否可用，不降級 widget 錯誤。

## 共用責任與目前邊界

NotebookAdapter 隔離 caller cfg／options 與核心工作輸入，不深拷貝大型 Result 或 Figure。Record-owned cfg／options 可被使用者刻意修改，不承諾完整不可變歷史。成功分析才成組提交 record 與 presentation handle。Run／load 成功清目前分析引用，失敗保留前次成功組；分析舊 source 不替換 last_run。失敗操作只收尾本次呈現，不關閉使用者持有的舊圖。Record 的 figures 只有具名 Mapping 與原生 Matplotlib 操作；presentation handle 另持有 Plots.release 責任。

明確 Notebook host 同步執行 caller 的操作，caller 負責順序。Host 在發布前完成
widget 初始化，普通更新與最後刷新都同步繪製，不等待 cell 結束才處理前端請求。
已有 manager 的 Figure 不能由另一 host 接管。完成操作不自動關閉圖，建立新操作也不關閉舊圖；caller 明確
呼叫 `Plots.release()` 才釋放 canvas／toolbar，原 Figure 仍可 `savefig`。直接建立
widget 而不登記 pyplot manager，避免 cell 結束時額外自動呈現。T1／GE／OneTone FluxDep convenience 使用此 host；其他實驗與舊互動分析入口仍待遷移。Host 不取代
那些入口的狀態與分析契約。

共用原始頻譜型別與座標整理位於 [`analysis/spectrum.py`](../analysis/spectrum.py)。Fluxdep 的共用 transition 型別、轉換及頻譜集合 I/O 位於 [`analysis/fluxdep`](../analysis/fluxdep/README.md) 的 `models.py`、`io.py`。共用 database search 位於 [`analysis/fluxdep/search.py`](../analysis/fluxdep/search.py)；Notebook 保留 `search_in_database` 組合入口與 `fit_spectrum` 微調。診斷圖由 [`plotting/fluxdep`](../plotting/fluxdep/README.md) 建立。

物理能譜等計算由 [`simulate/fluxonium`](../simulate/fluxonium/README.md) 提供，參數資料的 typed owner 見 [`resources`](../resources/README.md)。Notebook 可使用這些能力，也保留工作流程所需的專用計算。`TwoLinePicker` 的 Matplotlib rendering 目前仍在 [`analysis/fluxdep/line_picker.py`](../analysis/fluxdep/line_picker.py)，Notebook 的選點介面在 `analysis/fluxdep/interactive/`；此處只描述現有位置，不指定後續搬移。
