# `zcu_tools.experiment` — 實驗層

**Last updated:** 2026-10-06，native schema 公開投影

`experiment/` 定義實驗介面、Result 與保存資料之間的映射，以及實驗 cfg 的支援機制。這個位置不表示其現有程式碼均與 QICK 無關。`cfg_assembler.py` 從 `program.v2` 匯入 `ModuleCfgFactory`，`utils/sweep.py` 使用 `SweepCfg`。程式指令的建構與 acquire 由 [program/v2](../program/v2/README.md) 及 [experiment/v2/runtime](v2/runtime/README.md) 接合。

## 實驗介面與資料

- `records.py` 的 `RunRecord` 將 nullable cfg 與純 Result 配對。建立時複製 cfg，隔離 caller 之後對輸入 cfg 的修改；record 內 cfg 與 Result 仍可修改。`interfaces.py` 的 `RecordExperiment` 定義 run／save／load，`SynchronousExperiment` 另加同步 analyze，不要求 interactive 實驗提供 analyze。
- `base.py` 的 `PersistableExperiment` 透過 `AXES_SPEC` 提供 stateless `save(source, destination)`／`load(source)`，路徑使用本機 `Path`，公開介面不提供 server_ip／port。保存接明確 `RunRecord`，cfg 缺失時拒絕。載入回傳 `RunRecord`。Missing cfg 或非 envelope comment 使 cfg=None；recognized envelope 欄位型別錯誤報 ValueError。合法 envelope 的 cfg object 未通過 cfg model 時才 warning，並保留有效資料。核心不保存上次結果；NotebookExperiment 管理各實驗的便利引用，NotebookAdapter 只綁定可重用環境。
- `native_persistence.py` 的 `native_schemas` 由同一宣告投影出 datafile disk-unit schemas，供離線 caller 使用，不要求 caller 重組 axes／variable mapping。`save_run`／`load_run` 與 instance 同名入口使用顯式 AxesSpec／GroupedAxesSpec。框架把 Result 映射成 SI payload，再由 datafile 驗證與讀寫 native HDF5。Loader 核對 tag、cfg type 與 schema major，投影宣告的 variables 與 cfg，回傳 typed RunRecord 和歷史 RunSnapshot。Unknown cfg keys 留在 generic StoredRun，不塞進 typed cfg。現行 `save`／`load` 仍寫讀 Labber，不猜 native 格式。
- `context.py` 的 `RunContext` 提供單次 run 的 soc、soccfg、具名 devices、plots 與 cancel_signal。名稱到 driver 的對應在建立時固定，driver 仍由 caller 管理。每次 run 建立新 context、plots 與 `stop_signal.py` 的 StopSignal；硬體 handle 可重用，不把上一輪停止或錯誤狀態帶入下一輪。核心不管理前端 last_result 或呈現資源生命週期。
- `axes_spec.py` 用 `AxesSpec`、`Axis`、`ZSpec` 描述 Result 欄位如何對應檔案中的軸與資料 channel。`AxesSpec` 在宣告時核對 Result 的資料欄位，不要求 Result 包含 cfg。軸以 inner-first 排列；保存時套用宣告的單位比例，載入時還原成 Result 欄位。RunRecord 的 cfg 經 comment 保存，載入時由 cfg 型別驗證還原。`PersistableExperiment.save` 使用 caller 指定的路徑，不替 caller 選檔名。對包含多個 Data Variable 的單一 Result，`GroupedAxesSpec`／`VariableSpec` 描述 variable 與欄位的映射，由 typed builder 重建純 Result；`GroupedAxesSpec.save(source, destination)`／`load(source)` 與一般 persistence 一樣接明確 RunRecord／Path，cfg 缺失時拒絕保存，載入 comment 的錯誤分流與 single 入口一致，只有合法 envelope 的 cfg object 未通過 cfg model 時保留資料並警告。CPMG、Readout AutoOpt 與 JPA AutoOptimize 共用此 grouped record 入口。通用檔案格式與讀寫由 [datafile](../datafile/README.md) 負責；保存權責另見 [ADR-0063](../../../docs/adr/0063-persistence-ownership.md)。

## Cfg 與父層支援

- `cfg_model.py` 的 `ExpCfgModel` 繼承 package-root `zcu_tools.cfg_model.ConfigBase`，提供可選的 `dev` 欄位。`ConfigBase` 提供欄位驗證、`with_updates` 和 `to_dict`；`ExpCfgModel.validate_or_warn` 嘗試驗證載入的 cfg，失敗則發出警告並回傳 `None`。
- `cfg_assembler.py` 的 `assemble_experiment_cfg` 使用 caller 傳入的 module library 與 device snapshot，套用 overrides、組裝 device／module／單一 sweep 資料，再驗證成目標實驗 cfg。`make_cfg(raw_cfg, CfgModel, env)` 每次從 `CfgEnv.device_manager` 讀取當次資訊。`CfgEnv` 集中 md／ml／manager 引用，不解析 md expression，也不擁有資源關閉責任。這是量測 cfg 的組裝，不是 GUI 的編輯模型。
- `config.py` 目前提供繪圖尺寸 `figsize`；它不是 cfg model，也不執行 cfg 組裝。
- `utils/` 提供 cfg 到 datafile comment codec 的 mapping、device 設定／啟用與單一 sweep 格式整理。Comment envelope 的 JSON 編解碼與 Labber reader 的 labels／units／shape／numeric container 驗證由 datafile 擁有；experiment 保留 typed cfg validation、memory units 與 generated coordinate 語意。其中 device helper 接收顯式具名 driver mapping 與 StopSignal，先驗證整批名稱，再逐一 setup。只有 helper 呼叫 driver.setup 時才取 signal.event，低層 device 不依賴 experiment。Sweep helper 使用 program/v2 的 `SweepCfg`。`experiment.utils.make_sweep` 是 core 與 Notebook 共用的 sweep 建構入口；Notebook 不擁有此框架能力。它與 [v2/utils](v2/README.md#v2-工具與實驗輔助) 的硬體量化、SNR 和 tracker 工具分開。

## 子模組入口

Package root 保留公開 exports，只有存取對應名稱時才載入其 owner module。匯入 `cfg_editing` 不會因此載入 experiment base、device 或 datafile；直接使用 persistence exports 仍會載入其所需依賴。

- [v2](v2/README.md)：program/v2 的通用 runtime 與量化、SNR、tracker 工具。[runtime](v2/runtime/README.md) 說明 Schedule、buffer、ProgramBuilder 與 executor 的執行機制。具體實驗與前端附件由使用者套件提供，組合根注入 catalog，不由框架反向載入。
- [cfg_editing](cfg_editing/README.md)：Qt-free 的實驗側 program cfg 編輯支援；與執行前 cfg 組裝分工。
