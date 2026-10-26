# `zcu_tools.notebook`

**Last updated:** 2026-10-27

**讀者：** lab-author。Notebook 實驗入口、互動分析與支援模組導覽。

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

核心的 `AXES_SPEC=None` 時，存讀入口 raise `NotImplementedError`。Analyze 與 post-analyze 的基底簽名只宣告參數，未實作時同樣拒絕。

Caller 從實驗類別建立 options，例如 `T1Exp.Options(...)` 或 `GE_Exp.PostOptions(...)`。Options 由核心實驗類別提供。

每次包裝產生獨立的 `NotebookExperiment[CoreT]` 與 records。建構只綁定呼叫端提供的環境；裝置操作在 Run 或明確命令時執行。Run 需要 soc、soccfg 與 manager，無裝置時明確提供空 DeviceManager。每次 run 從綁定的 manager 取得一次 driver mapping，該次執行固定使用；同一 manager 的 registry 更新在下一次 run 生效。重新賦值 Notebook 變數不會重綁入口，換用另一組環境時由 caller 重建。

`NotebookAdapter()(core)` 可離線 load／analyze。同步 analyze 的 options 仍必傳，只有 source 可以省略。`analyze(options, *, source=None)` 與 `save(destination, *, source=None, ...)` 預設最近成功 run／load 的 last_run，沒有 record 就拒絕。處理歷史資料時顯式傳 source；分析歷史來源不改 last_run，也不改下一次 save 的預設來源。Save 只處理本機路徑，unique=True 才選新檔名並回傳實際 Path；`Experiment[Result, Cfg]` 宣告核心契約。包裝層將 `core.AXES_SPEC` 交給外部 `save_labber`；cfg 要求與禁止覆寫由 persistence 函式守住。

`load(Path(...))` 透過 `load_run_with_metadata(path, spec=core.AXES_SPEC)` 讀取與 GUI 相同的 `data.h5`，回傳 `RunRecord`。Single 與 grouped 資料共用 restore；歷史 cfg 不相容時保留有效資料、回傳 cfg=None 並發出 `CfgUnavailableWarning`。成功載入會清除目前分析與 acquisition evidence，失敗保留舊成果。只有 Labber 的舊資料必須先用既有 `migrations/` 工具離線轉成 native，不提供執行期 fallback。本機單檔 `save` 仍是明示的 Labber 匯出，不與 `load` 構成格式對稱。

`save_run(ctx, source=record, ...)` 是綁定 PointView 的 run 保存入口，與本機單檔 `save` 分開。Source 必須是此 adapter 最近成功 run 返回的同一 record。入口從 bound point 捕捉 cached context 與 storage capability，量測身分、時間與 provenance 沿 adapter 的 acquisition evidence。可選的 RoleView 必須來自同一 point，省略代表沒有角色對應。入口保存正規 `data.h5`、Labber 衍生輸出、具名 PNG／副本及窗格 JSON。`SavedRun.save()` 明確重存同一 run，固定輸出覆寫，Labber 每次取新檔；後續 adapter 操作不使此 handle 失效。

Run 命名沿 core.AXES_SPEC 的 `save_stem(SaveContext) -> str`。Caller 可在自己的 spec 指定 callback，框架不補任何名稱。
Context 包含 tag、run_id、point、roles、cfg values 與帶 timezone 的 saved_at。預設 stem 為 dotted tag、hyphen-joined roles 與完整 run_id，例如 `twotone.time_domain.t1_Q1-R1_20261027T120000Z-a1b2c3`。
Native 寫 `runs/<stem>/data.h5`，Labber 寫日期樹內的 `<stem>.hdf5`。只有 Labber 碰撞才加 numeric suffix；native 同名仍覆寫。
每次保存都呼叫 callback，自訂名若依保存時間改變，也會改變 native 目的地。Labber comment 同時保存歷史 snapshot.point 與 snapshot.roles。

`SavedRun.report` 與 `paths` 只表示最新一次保存結果。輸出或編碼失敗時，`PartialSaveError` 保留同一 SavedRun 及 report，caller 修正內容後可明確再存。Ledger 失敗直接傳播，不轉成 PartialSaveError，也不自動 retry。Figures／analysis 為 None 時保留上一組，空 mapping／tuple 清除該組。Run 身分、量測時間與 captured context 不因重存改變。

`SavedRun.accept(parameter=..., module=..., waveform=..., records=...)` 共用 entry 的寫回流程。Parameter 接受普通點分路徑的工作單位 JSON／complex 值，可用 `(value, metadata)` 以 mapping 整份替換該路徑的 opaque metadata，例如 `{"stderr": 0.8}`。只給 value 時保留既有 metadata；舊 `(value, float)` 形狀報錯。Module 與 waveform 接受 library key 到具名持久 Spec 的 mapping，不接受 raw dict。各類至少一個非空。Records 是具體元件名到 JSON 欄位 dict；流程只補 run_id、point、at 與 accepted source。Caller 提供的 dev 或其他儀器欄位原樣保留，缺 dev 也成功，資料一致性由 caller 保證。Notebook 不捕捉 alias，也不從 live device、cfg 或 snapshot 推論樣本值。Parameter／library 的 role 首段沿保存時固定的映射展開，records 不解析角色。Accepted 事件與 library provenance 的 kind 自動使用該 run 的實驗名稱。

流程先追加一筆 typed accepted 事件，再依目的地文件分組提交。Parameter 為 point 一組，module 與 waveform 共用 library 一組，records 逐筆提交。提交順序為 point、library、records，組內保留各 mapping 的迭代順序。Point 與 library 各使用一次 edit，任一項無效整組不寫，並停止後組。`WritebackApplyError` 保留 accepted `event_id`、已成功組的項目數 `applied`、逐項實際值的 `result.committed` 與原始 `__cause__`。已成功組及事件保留，不提供跨文件回復、retry 或恢復流程。單項 parameter／library setter 仍可獨立使用，accept 的整組提交不逐項呼叫它們。Ledger append 失敗在寫入值之前，錯誤原樣傳播。成功回傳 event ID。後續 adapter run 或保存其他 point 不改這個 handle 的 point、roles 與 run_id。GUI 使用同一 executor，proposal 與 applied 狀態仍由 GUI 管理。

Caller 明確提供 Notebook 自己的 bindings，RoleView 保留顯式 kind／via 契約。
GUI 的 `aliases.yaml` 與 Measure alias preset 由 GUI 使用。
GUI／MCP 的 temporary aliases 或 target 不改這個保存 handle。

## 工作家族

- [`adapter.py`](adapter.py)：共用環境綁定、每實驗 records 與操作收尾。每次 run 新建 RunContext、Plots 與 run 根 Scope，同步分析另建 Plots。入口借用 drivers，不管理其連線生命週期。
- [fluxdep](analysis/fluxdep/README.md)：通量依賴光譜的資料處理、擬合、圖表與互動選點。`fitting.py` 保留 `fit_spectrum` 與組合搜尋、診斷圖的 `search_in_database` 入口；數值搜尋由 analysis 擁有。
- [t1_curve](analysis/t1_curve/README.md) 與 [t2_curve](analysis/t2_curve/README.md)：各自保留 Notebook 的曲線分析、擬合與分階段工作流程。模型選擇與保存條件見各自的 README。
- [fit_tools](analysis/fit_tools/README.md)：提供 Notebook 分析專用的校正、資料接合、loss、weights 與溫度模型。
- [design](analysis/design/README.md)：評估模型參數、設計需求與候選組合，並分析 HFSS sweep 資料。
- [mist](analysis/mist/tool.py)：提供能量摺疊、不連續處理與碰撞遮罩的計算工具。
- [circuit_design](circuit_design/README.md)：提供 Qiskit Metal 電路幾何元件；相關 Notebook 展示電路建構及設計檔輸出。
一般 T1 使用 `nb_adapter(T1Exp())` 與核心的 typed config／options，不另提供扁平參數 wrapper。T1Result 只有資料；RunRecord 是保存與分析的 explicit 來源。Canonical load 可保留 cfg=None 的有效資料，T1 可離線分析，預設 saver 則拒絕缺 cfg 的來源。
- GE 的 run／FIT／save／load 使用 `nb_adapter(GE_Exp())` 與核心 typed cfg／options。Caller-owned `GEPostAnalyzer` 是獨立 post 工具，明確接收 `GEPrimaryRecord` 與 post options。Primary 是共用 AnalysisRecord；post record 從 primary 取得同一來源，保留其 calibration、options 與純圖。工具不讀 Adapter 的 current run／FIT，不重新 fitting。成功收尾後才發布 post record 與 presentation handle；失敗保留舊成果，舊原生圖仍可保存。
- Caller-owned `flux_dep.FluxDepAnalyzer.start(source, inputs, options)` 明確接收 opaque RunRecord 與獨立 FluxPickInputs，不從具體實驗結果抽取 spectra。Done 保存捕捉的 source、實際終態 FluxPickState、數值與純具名 `pick` Figure。工具另持有 Plots presentation handle。Cancel 或失敗保留舊成果；Adapter 的 run／load 不清除此工具的分析。Notebook 拖曳預覽與 measure click-follow frontend 各自保留 gesture／lifecycle。
- [`utils.py`](utils.py)：提供圖檔保存與設備資訊等 Notebook 輔助函式。Sweep 建構使用 `experiment.utils.make_sweep`，與核心實驗共用框架入口。`dump_device_info` 與 `reconnect_devices` 接收 caller 的 DeviceManager，不查全域 registry。
- [`plotting.py`](plotting.py)：`NotebookPlotHost` 實作共用 `PlotHost`，直接以 ipympl widget 呈現原生 Figure。不登記 pyplot manager，也不切換全域 backend。普通圖與 liveplot 的呈現時機由 `Plots` 控制，host 不偵測 browser 是否可用，不降級 widget 錯誤。

## 共用責任與目前邊界

Caller 直接建立實驗的 typed cfg，再傳給 `NotebookExperiment.run(cfg)`。`experiment.cfg_assembler` 只提供 Run mapping 的共用值型別 `ExperimentScalar` 與 `ExperimentCfgValue`。需要 library 成品時，caller 可從固定 `LibrarySnapshot` 查詢；切換參數或 library 來源後，由 caller 重新捕捉所需 snapshot 並重建 cfg。GUI 的 schema／expression lowering 由 GUI 表單處理，Notebook 入口直接使用 caller 提供的 cfg。

Cfg.dev 只保存角色到 device 名稱的對應，使用 flux 的 cfg 必填 flux_unit，值為 A、V 或 native。Caller 必須明示 unit，不提供隱含 default。Run 不預查 cfg.dev 名稱是否存在，cfg 格式與必填 unit 的驗證仍保留。Core 實際以 `context.devices[cfg.dev[role]]` 選 driver 時，缺名即報錯並指名 device。Core 透過實驗自己的 helpers 設 flux／RF／output，不把 DeviceInfo 當設定命令，也不對未使用的 device 套 setup patch。單位換算由 caller 負責，RF helpers 接 Hz／dBm。RunContext 不提供 setters。

Caller 負責明確的 setup 與資源關閉。Run 不重新組裝或刷新 cfg、不重連或自動補裝置。建立 cfg 時的裝置讀取不能當成 Run snapshot。

Notebook Run 在使用者的程序內同步執行，資源由 caller 擁有。入口只借用 drivers，不新增 lease、lock 或 registry。Driver mapping 只固定名稱到 instance，不是資源 lease。Hardware lease 後的 snapshot 時機只適用 GUI。

Notebook 在建立 RunContext 後、core.run 前，對全部已連線 device 各讀一次 get_info 作為 snapshot。這一步只讀不寫。任一台失敗拒絕 Run，不進 core、不以 cache 補值；無 device 記 `{}`。Snapshot 表示實驗前狀態，後續 sweep 不覆寫。保存沿 RunMetadata.device_snapshot 到既有 data.h5 `/device`，Save 不重讀硬體或取下一輪值。

`acquisition_for(source)` 只接受此 NotebookExperiment 最近成功 run 的同一 RunRecord，回傳隔離的 run identity、UTC 開始／結束時間、complete／stopped 與 software provenance。Run 在 core acquire 前捕捉 software evidence，正常返回且 scope.stop 錯誤檢查通過後結束計時，不包含 plots.finish、analyze 或 save。Presentation 成功後才成組發布 record 與 evidence。Analyze 與所有失敗保留 evidence，下一次成功 run 取代它，成功 load 清除它。Loaded、手組、另一 adapter 或過期 record 不能取得此 acquisition evidence。Schedule 會接住 scope 內的一次 KeyboardInterrupt，保留 buffer 與 interrupted signal。Notebook 將這種正常返回發布為 stopped record，已完成點與其餘 NaN 可沿 save_run 保存。Adapter 不另加 cancellation 入口。Completion 依 `scope.stop.partial_result` 判定，子 scope 取消即使不設 root event，仍標為 stopped。

NotebookExperiment 隔離 caller cfg／options 與核心工作輸入，不深拷貝大型 Result 或 Figure。Record-owned cfg／options 可被使用者刻意修改，不承諾完整不可變歷史。成功分析才成組提交 record 與 presentation handle。Run 在提交前檢查 scope.stop 的失敗原因，completion 讀取 partial_result，不只讀根的取消 bit。Failed 仍拋錯，interrupted 視為使用者停止，與無錯誤的 stopped partial 一樣可提交。Schedule 外的 KeyboardInterrupt（例如實驗 helper 的儀器操作）直接傳播，保留舊成果。GUI／MCP 的 raise_if_error 政策不變。Run／load 成功清目前分析引用，失敗保留前次成功組；分析舊 source 不替換 last_run。Notebook 的 run、同步分析與 GE post 共用失敗呈現的收尾，保留 producer 與 cleanup 錯誤，只釋放本次呈現，不關閉使用者持有的舊圖。Record 的 figures 只有具名 Mapping 與原生 Matplotlib 操作；presentation handle 另持有 Plots.release 責任。

明確 Notebook host 同步執行 caller 的操作，caller 負責順序。Host 在發布前完成
widget 初始化，普通更新與最後刷新都同步繪製，不等待 cell 結束才處理前端請求。
已有 manager 的 Figure 不能由另一 host 接管。完成操作不自動關閉圖，建立新操作也不關閉舊圖；caller 明確
呼叫 `Plots.release()` 才釋放 canvas／toolbar，原 Figure 仍可 `savefig`。直接建立
widget 而不登記 pyplot manager，避免 cell 結束時額外自動呈現。共用 NotebookAdapter 與 FluxDepAnalyzer 使用此 host。Host 不取代
這些入口的狀態與分析契約。

共用原始頻譜型別與座標整理位於 [`analysis/spectrum.py`](../analysis/spectrum.py)。Fluxdep 的共用 transition 型別、轉換及頻譜集合 I/O 位於 [`analysis/fluxdep`](../analysis/fluxdep/README.md) 的 `models.py`、`io.py`。共用 database search 位於 [`analysis/fluxdep/search.py`](../analysis/fluxdep/search.py)；Notebook 保留 `search_in_database` 組合入口與 `fit_spectrum` 微調。診斷圖由 [`plotting/fluxdep`](../plotting/fluxdep/README.md) 建立。

物理能譜等計算由 [`simulate/fluxonium`](../simulate/fluxonium/README.md) 提供，參數資料的 typed owner 見 [`resources`](../resources/README.md)。Notebook 可使用這些能力，也保留工作流程所需的專用計算。`TwoLinePicker` 的 Matplotlib rendering 目前仍在 [`analysis/fluxdep/line_picker.py`](../analysis/fluxdep/line_picker.py)，Notebook 的選點介面在 `analysis/fluxdep/interactive/`；此處只描述現有位置，不指定後續搬移。
