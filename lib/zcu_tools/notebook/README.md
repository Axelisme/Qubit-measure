# `zcu_tools.notebook`

**Last updated:** 2026-10-02, Options namespace

`zcu_tools.notebook` 提供 Notebook 逐步探索時使用的互動入口、顯示與 widgets，也保留工作流程專用的分析支援。Notebook 工作流程可組合計算與人工確認，不等於 GUI 的量測 session 或狀態管理。實際操作與結果解讀見 [Notebook 內容入口](../../../notebook_md/README.md)；這裡說明支援程式的位置。

## 共用實驗入口

`NotebookAdapter` 先綁定 soc、soccfg、DeviceManager 與 host 引用，再包裝 caller 建立的 experiment instance：

```python
nb_adapter = NotebookAdapter(
    soc=soc, soccfg=soccfg, device_manager=device_manager
)
exp = nb_adapter(T1Exp())
_ = exp.run(cfg)
_ = exp.analyze(T1Exp.Options(skip=3))
saved = exp.save(destination, unique=True)
```

Caller 從實驗類別建立 options，例如 `T1Exp.Options(...)` 或 `GE_Exp.PostOptions(...)`，不必再匯入各自的 Options 型別。NotebookExperiment 不轉送核心屬性，所以不是 `exp.Options(...)`。

每次包裝產生獨立的 `NotebookExperiment[CoreT]`，不在環境入口共用 records。建構不查裝置、不連線、不 setup。Run 需要 soc、soccfg 與 manager，無裝置時明確提供空 DeviceManager。每次 run 從綁定的 manager 取得一次 driver mapping，該次執行固定使用；同一 manager 的 registry 更新在下一次 run 生效。重新賦值 Notebook 變數不會重綁入口，換用另一組環境時由 caller 重建。

`NotebookAdapter()(core)` 可離線 load／analyze。同步 analyze 的 options 仍必傳，只有 source 可以省略。`analyze(options, *, source=None)` 與 `save(destination, *, source=None, ...)` 預設最近成功 run／load 的 last_run，沒有 record 就拒絕。處理歷史資料時顯式傳 source；分析歷史來源不改 last_run，也不改下一次 save 的預設來源。Save 只處理本機路徑，unique=True 才選新檔名並回傳實際 Path；cfg 要求與禁止覆寫由核心 saver 守住。

## 工作家族

- [`adapter.py`](adapter.py)：共用環境綁定、每實驗 records 與操作收尾。每次 run 新建 RunContext、Plots 與 StopSignal，同步分析另建 Plots。入口借用 drivers，不管理其連線生命週期。
- [fluxdep](analysis/fluxdep/README.md)：通量依賴光譜的資料處理、擬合、圖表與互動選點。`fitting.py` 保留 `fit_spectrum` 與組合搜尋、診斷圖的 `search_in_database` 入口；數值搜尋由 analysis 擁有。
- [t1_curve](analysis/t1_curve/README.md) 與 [t2_curve](analysis/t2_curve/README.md)：各自保留 Notebook 的曲線分析、擬合與分階段工作流程。模型選擇與保存條件見各自的 README。
- [fit_tools](analysis/fit_tools/README.md)：支援 Notebook 分析中的校正、資料接合、loss、weights 與溫度模型；不把這些能力一概視為共用分析核心。
- [design](analysis/design/README.md)：評估模型參數、設計需求與候選組合，並分析 HFSS sweep 資料。
- [mist](analysis/mist/tool.py)：提供能量摺疊、不連續處理與碰撞遮罩的計算工具。
- [circuit_design](circuit_design/README.md)：提供 Qiskit Metal 電路幾何元件；相關 Notebook 展示電路建構及設計檔輸出。
一般 T1 使用 `nb_adapter(T1Exp())` 與核心的 typed config／options，不另提供扁平參數 wrapper。T1Result 只有資料；RunRecord 是保存與分析的 explicit 來源。Canonical load 可保留 cfg=None 的有效資料，T1 可離線分析，預設 saver 則拒絕缺 cfg 的來源。
- GE 的 run／FIT／save／load 使用 `nb_adapter(GE_Exp())` 與核心 typed cfg／options。[`experiments/ge.py`](experiments/ge.py) 的 `GEPostAnalyzer` 是獨立 post 工具，明確接收 `GEPrimaryRecord` 與 post options。Primary 是共用 AnalysisRecord；post record 從 primary 取得同一來源，保留其 calibration、options 與純圖。工具不讀 Adapter 的 current run／FIT，不重新 fitting。成功收尾後才發布 post record 與 presentation handle；失敗保留舊成果，舊原生圖仍可保存。
- [`experiments/flux_dep.py`](experiments/flux_dep.py)：`FluxDepAnalyzer.start(source, options)` 明確接收 OneTone 或 TwoTone 通量光譜的 RunRecord，建立選線 widgets 與可拖曳的預覽。泛型保留 cfg／result 的來源型別，預設為 OneTone；TwoTone caller 明確指定 `FluxDepAnalyzer[FreqFluxCfg, FreqFluxResult]`。工具不綁定 core 或 Adapter 的目前來源。Done 保存捕捉的 source、實際終態 FluxPickState、數值與純具名 `pick` Figure。工具另持有 Plots presentation handle。Cancel 或失敗保留舊成果；Adapter 的 run／load 不清除此工具的分析。預覽與成果圖分開。
- [`utils.py`](utils.py)：提供 sweep、圖檔保存與設備資訊等 Notebook 輔助函式。`dump_device_info` 與 `reconnect_devices` 接收 caller 的 DeviceManager，不查全域 registry。
- [`plotting.py`](plotting.py)：`NotebookPlotHost` 實作共用 `PlotHost`，直接以 ipympl widget 呈現原生 Figure。不登記 pyplot manager，也不切換全域 backend。普通圖與 liveplot 的呈現時機由 `Plots` 控制，host 不偵測 browser 是否可用，不降級 widget 錯誤。

## 共用責任與目前邊界

Cfg 組裝使用 `experiment.cfg_assembler.CfgEnv(md, ml, device_manager)`。Notebook 顯式呼叫 `make_cfg(raw_cfg, CfgModel, env, overrides=...)` 時讀取當次裝置資訊，失敗直接報錯；底層 assembler 與 GUI 仍使用 caller 給定的 snapshot。CfgEnv 不執行 setup、不解析 md expression，也不關閉資源。切換 md／ml 後重新建立 CfgEnv。Run 不重做 make_cfg、不刷新 cfg、不重連或自動補裝置。Driver mapping 只固定名稱到 instance，不是資源 lease，driver guards 與既有 owner 不變。

NotebookExperiment 隔離 caller cfg／options 與核心工作輸入，不深拷貝大型 Result 或 Figure。Record-owned cfg／options 可被使用者刻意修改，不承諾完整不可變歷史。成功分析才成組提交 record 與 presentation handle。Run 在提交前檢查 StopSignal 的失敗原因，failed／interrupted 會拋錯；無錯誤的 stopped partial 仍可提交。Run／load 成功清目前分析引用，失敗保留前次成功組；分析舊 source 不替換 last_run。Notebook 的 run、同步分析與 GE post 共用失敗呈現的收尾，保留 producer 與 cleanup 錯誤，只釋放本次呈現，不關閉使用者持有的舊圖。Record 的 figures 只有具名 Mapping 與原生 Matplotlib 操作；presentation handle 另持有 Plots.release 責任。

明確 Notebook host 同步執行 caller 的操作，caller 負責順序。Host 在發布前完成
widget 初始化，普通更新與最後刷新都同步繪製，不等待 cell 結束才處理前端請求。
已有 manager 的 Figure 不能由另一 host 接管。完成操作不自動關閉圖，建立新操作也不關閉舊圖；caller 明確
呼叫 `Plots.release()` 才釋放 canvas／toolbar，原 Figure 仍可 `savefig`。直接建立
widget 而不登記 pyplot manager，避免 cell 結束時額外自動呈現。共用 NotebookAdapter 與 FluxDepAnalyzer 使用此 host。Host 不取代
這些入口的狀態與分析契約。

共用原始頻譜型別與座標整理位於 [`analysis/spectrum.py`](../analysis/spectrum.py)。Fluxdep 的共用 transition 型別、轉換及頻譜集合 I/O 位於 [`analysis/fluxdep`](../analysis/fluxdep/README.md) 的 `models.py`、`io.py`。共用 database search 位於 [`analysis/fluxdep/search.py`](../analysis/fluxdep/search.py)；Notebook 保留 `search_in_database` 組合入口與 `fit_spectrum` 微調。診斷圖由 [`plotting/fluxdep`](../plotting/fluxdep/README.md) 建立。

物理能譜等計算由 [`simulate/fluxonium`](../simulate/fluxonium/README.md) 提供，參數資料的 typed owner 見 [`resources`](../resources/README.md)。Notebook 可使用這些能力，也保留工作流程所需的專用計算。`TwoLinePicker` 的 Matplotlib rendering 目前仍在 [`analysis/fluxdep/line_picker.py`](../analysis/fluxdep/line_picker.py)，Notebook 的選點介面在 `analysis/fluxdep/interactive/`；此處只描述現有位置，不指定後續搬移。
