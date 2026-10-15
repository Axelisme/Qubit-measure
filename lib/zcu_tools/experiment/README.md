# `zcu_tools.experiment` — 實驗層

**Last updated:** 2026-10-16，FTF 環境與固定單位 flux helper

`experiment/` 定義實驗介面、Result 與保存資料之間的映射，以及實驗 cfg 的支援機制。這個位置不表示其現有程式碼均與 QICK 無關。`cfg_assembler.py` 只提供 Run mapping 的共用值型別，`utils/sweep.py` 使用 `program.v2` 的 `SweepCfg`。程式指令的建構與 acquire 由 [program/v2](../program/v2/README.md) 及 [experiment/v2/runtime](v2/runtime/README.md) 接合。

## Explicit simulation environments

`simulation.py` provides `SimulationEnvironment`, `white_noise` and
`load_environment`. A trusted Python file defines zero-argument `build_environment`.
It returns a bound SoC, its matching compile config, the exact populated
`DeviceManager` and a public description. GUI and Notebook share this bundle;
there is no controller argument, registry discovery or runtime mode switch.
Successful builder return transfers cleanup to the host. Read the
[acquisition environment contract](../program/v2/simulation/README.md) before
writing a builder or numeric source. Public Run metadata contains only the
optional opaque `simulation_id`; private replay inputs remain on the binding.

[`ftf`](ftf/README.md) composes a named YAML sample, fixed V/A memory devices,
fixed-coordinate joint circuits, explicit evolution/population conversion and the
single-receiver ADC. It uses the same bundle and acquisition contracts, without
adding GUI physics or exposing state labels to ordinary Run metadata.

## 實驗介面與資料

- `records.py` 的 `RunRecord` 將 nullable cfg 與純 Result 配對。建立時複製 cfg，隔離 caller 之後對輸入 cfg 的修改；record 內 cfg 與 Result 仍可修改。`interfaces.py` 的 `RecordExperiment` 定義 run／save／load，`SynchronousExperiment` 另加同步 analyze，不要求 interactive 實驗提供 analyze。
- `base.py` 的 `PersistableExperiment` 透過 `AXES_SPEC` 提供 stateless `save(source, destination)`／`load(source)`，路徑使用本機 `Path`，公開介面不提供 server_ip／port。保存接明確 `RunRecord`，cfg 缺失時拒絕。載入回傳 `RunRecord`。Missing cfg 或非 envelope comment 使 cfg=None；recognized envelope 欄位型別錯誤報 ValueError。合法 envelope 的 cfg object 未通過 cfg model 時才 warning，並保留有效資料。核心不保存上次結果；NotebookExperiment 管理各實驗的便利引用，NotebookAdapter 只綁定可重用環境。
- `native_persistence.py` 的 `prepare_run_data` 將明確 RunRecord/spec 映射為 `PreparedRunData`，提供同源 cfg 與 SI payload 給 native、Labber writer。Preparation 不做 I/O，保留 raw comment、single/grouped representation，驗證 typed cfg/result、schema 與 shape，不增加 Labber common-grid 限制。`PersistableExperiment.persistence_spec()` 公開 instance 的保存宣告，缺宣告明確失敗。`load_run_with_metadata` 讀回 typed record 與完整歷史 RunMetadata，延續 run 身分、時間、completion、context 與 provenance；舊 `load_run` 仍只回 snapshot。
- `native_persistence.py` 的 `native_schemas` 由同一宣告投影出 datafile disk-unit schemas，供離線 caller 使用，不要求 caller 重組 axes／variable mapping。`save_run`／`load_run` 與 instance 同名入口使用顯式 AxesSpec／GroupedAxesSpec。框架把 Result 映射成 SI payload，再由 datafile 驗證與讀寫 native HDF5。Loader 核對 tag、cfg type 與 schema major，投影宣告的 variables 與 cfg，回傳 typed RunRecord 和歷史 RunSnapshot。Unknown cfg keys 留在 generic StoredRun，不塞進 typed cfg。明示的本機單檔 `save`／`load` 保留 Labber API，不猜 native 格式；GUI Load Data 使用 native `load_run_with_metadata`，bound-point 保存使用共用 run saver。
- `run_saving.py` 的 `save_artifacts` 編排一次保存。正規資料寫入 `runs/<run_id>/data.h5`，每次另產 Labber 新檔。同 run 重存覆寫 native、PNG 與窗格 JSON，不覆寫 Labber。DataMember 共用同一 PreparedRunData，FileMember 交原生 Figure 或 AnalysisDocument，每成員編碼一次，對等路徑共用 bytes。輸出與編碼失敗各自隔離，每次呼叫回傳最新 SaveReport 並追加 saved 事件。Ledger 失敗直接傳播，已寫檔案保留，不 retry 或 rollback。`PersistableRecordExperiment` 結合 record core 與 persistence_spec，供正式 Notebook caller 保留 cfg／Result 配對。
- `acquisition.py` 集中 adapter 的 `AcquisitionEvidence`、UTC run identity 與兩種 capture。
  `capture_software_provenance` 只讀 package／Git／host 與已有的 SoC cached config，不查儀器。
  `capture_device_snapshot(devices)` 由 Run owner 在下述 checkpoint 呼叫，讀取全部 connected drivers 的設定並轉成隔離 JSON。
  量測時間和 completion 由前端 adapter 在 acquire 邊界記錄，不由 core 或 datafile writer 擷取。Evidence 複製版本 mapping，獨立於可修改的 RunRecord cfg／Result。
- Device snapshot 由兩個 Run 建立路徑各捕捉一次。GUI 在 hardware lease 後、adapter.run 前讀取。Notebook 同步執行，資源由 caller 擁有，不新增 lease；它在建立 RunContext 後、core.run 前讀取。兩路對全部已連線 device 各讀一次 get_info，只讀不寫。任一讀取失敗拒絕 Run，不進 adapter/core，不以 cached info 補值；無 device 記 `{}`。Snapshot 隨當次 Run 保存，後續 sweep 或 Save 不重讀、不覆寫。
- `context.py` 的 `RunContext` 提供單次 run 的 soc、soccfg、具名 devices、plots 與 cancel_signal。名稱到 driver 的對應在建立時固定，driver 仍由 caller 管理。RunContext 不提供儀器 setters；core 以 `context.devices[cfg.dev[role]]` 取得 driver，再使用實驗層 helper。每次 run 建立新 context、plots 與 `stop_signal.py` 的 StopSignal；硬體 handle 可重用，不把上一輪停止或錯誤狀態帶入下一輪。核心不管理前端 last_result 或呈現資源生命週期。
- `axes_spec.py` 用 `AxesSpec`、`Axis`、`ZSpec` 描述 Result 欄位如何對應檔案中的軸與資料 channel。`AxesSpec` 在宣告時核對 Result 的資料欄位，不要求 Result 包含 cfg。軸以 inner-first 排列；保存時套用宣告的單位比例，載入時還原成 Result 欄位。RunRecord 的 cfg 經 comment 保存，載入時由 cfg 型別驗證還原。`PersistableExperiment.save` 使用 caller 指定的路徑，不替 caller 選檔名。對包含多個 Data Variable 的單一 Result，`GroupedAxesSpec`／`VariableSpec` 描述 variable 與欄位的映射，由 typed builder 重建純 Result；`GroupedAxesSpec.save(source, destination)`／`load(source)` 與一般 persistence 一樣接明確 RunRecord／Path，cfg 缺失時拒絕保存，載入 comment 的錯誤分流與 single 入口一致，只有合法 envelope 的 cfg object 未通過 cfg model 時保留資料並警告。CPMG、Readout AutoOpt 與 JPA AutoOptimize 共用此 grouped record 入口。通用檔案格式與讀寫由 [datafile](../datafile/README.md) 負責；保存權責另見 [ADR-0063](../../../docs/adr/0063-persistence-ownership.md)。

## Cfg 與父層支援

- `cfg_model.py` 的 `ExpCfgModel` 繼承 package-root `zcu_tools.cfg_model.ConfigBase`，不宣告 `dev`。
  需要 device 的實驗 cfg 自行宣告角色到 device 名稱的 mapping；只有確實可選的實驗才宣告 optional dev。Dev 不保存 DeviceInfo、設定 patch 或 snapshot。
  使用 flux 的實驗 cfg 必填 `flux_unit`，值為 A、V 或 native，core 不提供隱含 default。
  Run 不預查 cfg.dev 名稱是否存在。Core 實際取 `context.devices[cfg.dev[role]]` 時，缺名即報錯並指名 device，不設定未使用的 device。Cfg 格式與必填 unit 的驗證仍保留。
  `ConfigBase` 提供欄位驗證、`with_updates` 和 `to_dict`；`ExpCfgModel.validate_or_warn` 嘗試驗證載入的 cfg，失敗則發出警告並回傳 `None`。
- `cfg_assembler.py` 提供 `ExperimentScalar` 與 `ExperimentCfgValue`，描述 Run 接縫的 mapping 值型別。GUI Form 在 accept 前完成 module materialization 與 sweep 轉換；BaseAdapter 使用 Form 的 core cfg 型別驗證 accept 的 dump，不再重組 module 或 sweep。Notebook caller 直接提供 typed cfg。
- `config.py` 目前提供繪圖尺寸 `figsize`；它不是 cfg model，也不執行 cfg 組裝。
- `utils/` 提供 cfg 到 datafile comment codec 的 mapping、實驗層 device helpers 與單一 sweep 格式整理。Comment envelope 的 JSON 編解碼與 Labber reader 的 labels／units／shape／numeric container 驗證由 datafile 擁有；experiment 保留 typed cfg validation、memory units 與 generated coordinate 語意。Device helpers 的角色與單位契約見下節。Sweep helper 使用 program/v2 的 `SweepCfg`。`experiment.utils.make_sweep` 是 core 與 Notebook 共用的 sweep 建構入口；Notebook 不擁有此框架能力。它與 [v2/utils](v2/README.md#v2-工具與實驗輔助) 的硬體量化、SNR 和 tracker 工具分開。

## 實驗角色與 device 操作

`utils/device.py` 的 helpers 取代 setup_devices 與 set_*_in_dev_cfg。DeviceInfo 是資訊，不再提供 set_flux／set_freq／set_power／set_output 控制命令。Device 層只提供儀器本身的操作，不知道 flux 或 RF 角色；實驗層選 driver、明示單位與操作用途。

- `check_flux_unit(cfg: BaseDeviceInfo, unit: str) -> None` 是純檢查，只比 YOKO／Fake Info 儲存的 value_unit。YOKO Info 驗 current／A 與 voltage／V，Fake 為建立時固定的 native、V 或 A。Mismatch 或契約外 unit 報 ValueError，其他 Info 報 TypeError 並指名不支援 flux 設定的類別。不換算、不改 cfg，不讀寫硬體或 cache。
- `set_flux(device, value, *, unit, stop, progress=False)` 委派同一 checker，保既有 mode 查詢、stop、progress 與 driver safety，不新增 get_info 或其他硬體查詢。接受 A、V 或 native。YOKO 的 A 要求 current 模式並呼叫 set_current，V 要求 voltage 模式並呼叫 set_voltage。Fake 比對 driver 實際 get_info() 回報的固定單位，再呼叫 set_value，不新造 native Info。單位或模式不符在寫入前拒絕。Helper 不換算、不切模式、不改 rampstep、不自動開 output，YOKO ramp 使用同一 stop 並沿既有安全限制。
- `set_rf_frequency(device, freq_hz)` 呼叫 SGS100A 的 set_frequency 或 Fake 的 set_value。單位為 Hz，MHz 等換算由 caller 在呼叫處完成。
- `set_rf_power(device, power_dbm)` 呼叫 SGS100A 的 set_power 或 Fake 的 set_value，單位為 dBm。
- `set_output(device, "on" | "off")` 呼叫支援 driver 的 set_output。

Helpers 對其他 device 種類報 TypeError，driver 錯誤原樣傳播。FakeDevice 維持單一 value，不新增 RF 欄位、protocol 或 registry；模擬頻率與功率仍寫同一值。

Workflow 的同步 `get_device_cfg(name)` 與完整 cfg `setup_device(name, cfg)` 由宿主提供。Zcu_lab 工具組合讀取、`check_flux_unit` 與欄位修改，framework 不合併部分 cfg，也不新增通用 value＋unit effect。Measure MCP 的 partial `device_setup` 是另一個有界工具，見 ADR-0068。

## 子模組入口

Package root 保留公開 exports，只有存取對應名稱時才載入其 owner module；直接使用 persistence exports 仍會載入其所需依賴。

GUI 的 Library、Inspect 與 module writeback 共用 `gui.form` 與 `gui.form_adapter`，直接提交持久 Spec，不經 experiment 的第二套編輯樹。ModuleWriteback 的 proposal 是 ModuleSpec；experiment 仍只接收執行用 typed cfg。跨宿主契約見 ADR-0065、ADR-0008。

- [v2](v2/README.md)：program/v2 的通用 runtime 與量化、SNR、tracker 工具。[runtime](v2/runtime/README.md) 說明 Schedule、buffer、ProgramBuilder 與 executor 的執行機制。具體實驗與前端附件由使用者套件提供，組合根注入 catalog，不由框架反向載入。
