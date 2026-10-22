# `zcu_tools.experiment` — 實驗層

**Last updated:** 2026-10-22

**讀者：** framework-dev。實驗介面、Run 資料、持久化映射與裝置操作邊界。

`experiment/` 定義實驗介面、Result 與保存資料之間的映射，以及實驗 cfg 的支援機制。這個位置不表示其現有程式碼均與 QICK 無關。`cfg_assembler.py` 提供 Run mapping 的共用值型別，`utils/sweep.py` 使用 `program.v2` 的 `SweepCfg`。程式指令的建構與 acquire 由 [program/v2](../program/v2/README.md) 及 [experiment/v2/runtime](v2/runtime/README.md) 接合。

## Shared async script host

`script_host.ScriptHost` 建構時不啟動 thread 或 loop。`run(coroutine)` 在呼叫它的 worker 執行一次作者 coroutine，完成 finally 與 executor drain 後返回或 raise，不能重用。`await call(fn)` 在 executor 執行 `fn(CallContext)`，回 `CallResult(result, epoch, partial_result)`。Context 提供本次 operation 的 StopSignal child 與起跑 epoch，停止的 token 仍啟動 fn，由 owner 合作返回 partial。作者 Task 不會被 cancel，finally 留在同一 worker。

`await ask(question)` 以唯一 request id 等待具名 Answer，`answer(request_id, answer)` 驗證當前問題；錯誤、重複或遲到答案不消耗新問題。Host script cancel 以內部 None 釋放 Ask Future，recipe 便利層轉成普通 RecipeCancelled，作者 ask 不回 Optional。

`control(fn)` 把數微秒級的 memory-only callback 排入與 cancel／pause／resume／answer 相同的 loop 序列，呼叫端等待回傳值或原例外。Fn 不 await、不阻塞、不 I/O，也不重入作者程式。Worker 上 inline 執行，foreign thread 經 call_soon_threadsafe 與 acknowledgement；run 前或 loop 關閉後 raise RuntimeError。`cancel(scope="script"|"run")` 只設 token，acknowledgement 不表示 operation 已 drain。Recipe runtime 另用 control 原子檢查當前 tab.run token，analysis／Ask／閒置時 run cancel 是 no-op。`close()` 合作 drain，off-worker 等到 coroutine／finally／executor 完成，worker 自己呼叫時只請求停止，不自等。

Pause 只停止目前 operation、增加 epoch 並關 gate，不停止 iteration。Resume 只開 gate。CallResult 固定起跑 epoch 與 fn 完成時的 stop bit。Workflow env 決定是否丟棄 partial 並重跑相同 request；host 不重跑外部效果。Commit 等 gate 後更換 iteration child。Script cancel 打開 gate，使作者可收尾。

`stop_signal.StopSignal.child()` 在出生時繼承停止與錯誤，set 往下傳。`set_error()` 只在本 scope 記第一個錯誤並停止自己與子孫，不改祖先或兄弟的停止與錯誤狀態。`release()` 移除已完成 scope。原 `script_driver.ScriptDriver` 暫只服務尚未遷移的 workflow；其直接 `cancel()` 保留既有邊界語意，不提供控制訊息 inbox。

## Explicit simulation environments

`simulation.py` provides `SimulationEnvironment`, `white_noise` and
`load_environment`. A trusted Python file defines zero-argument `build_environment`.
It returns a bound SoC, its matching compile config, the exact populated
`DeviceManager` and a public description. GUI and Notebook share this bundle;
there is no controller argument, registry discovery or runtime mode switch.
`create_device(type_name, address)` retrieves explicitly addressed startup
FakeDevices from their process-wide class container, preserving model bindings
and settings after disconnect/name changes. The environment keeps only modeled
address metadata, not another instance registry. Unknown types/addresses fail
explicitly. Anonymous startup devices have no reconnect identity.
Successful builder return transfers cleanup to the host. Read the
[acquisition environment contract](../program/v2/simulation/README.md) before
writing a builder or numeric source. Public Run metadata contains only the
optional opaque `simulation_id`; private replay inputs remain on the binding.
WhiteNoise numeric code lives in `simulate.white_noise`. This composition root
wraps its statistics and complete raw source in program's `IntegratedRound`.
Decimated acquisition keeps per-execution `RoundSignals`. ADC and white_noise
structurally satisfy program's `RawShots` without importing program. Processing
or a getter requests shots from a separate retained identity. Direct means keep
their existing moment calculation and RNG stream.

[`ftf`](ftf/README.md) composes a named YAML sample, fixed V/A memory devices,
fixed-coordinate joint circuits, explicit evolution/population conversion and the
single-receiver ADC. It uses the same bundle and acquisition contracts, without
adding GUI physics or exposing state labels to ordinary Run metadata.

[`single_fluxonium`](single_fluxonium/README.md) plans complete microwave/fast-flux
timelines and assembles a one-source mock environment. Its source composes DC-basis
population conversion, one-cavity ADC statistics, deferred raw shots and decimated
traces. Hosts can call the optional `SimulationEnvironment.close_backend` after
acquisition drains; device cleanup remains separate.

`readout_plan.py` owns shared QICK drive/readout waveform conversion without
settings or circuit physics. `drive_track` detaches microwave slots with the
same support, carrier-phase and envelope semantics for both environments. It partitions terminal readout and earlier cavity
drives, detaches envelopes, deduplicates pulse/grid contents and converts
common-origin carrier phases to pulse-local phases. Environment planners supply
`ReadoutDrive` calibration and retain responsibility for receiver binding,
physical models, evolution and sampling.

## 實驗介面與資料

- `records.py` 的 `RunRecord` 將 nullable cfg 與純 Result 配對。可選 `labber_metadata` 保存載入檔案的 comment、tags、project、user 與 creation_time；未捕捉時為 None。Typed `restore_run`、`load_run`、`load_run_with_metadata` 與 core `from_stored` 都沿此欄位保留 metadata，不改各入口原有 tuple 含義。建立時複製 cfg 與 Labber metadata，隔離 caller 之後對輸入的修改；record 內 cfg、Result 與 metadata 仍可修改。`interfaces.py` 的 `RecordExperiment` 定義 run／save／load，`SynchronousExperiment` 另加同步 analyze，不要求 interactive 實驗提供 analyze。
- `base.py` 的 `PersistableExperiment` 透過 `AXES_SPEC` 提供 stateless `save(source, destination)`／`load(source)`，路徑使用本機 `Path`，公開介面不提供 server_ip／port。保存接明確 `RunRecord`，cfg 缺失時拒絕。載入回傳 `RunRecord`。`save` 是單檔 Labber 匯出，`load` 讀 native `data.h5` 並透過公開 `from_stored` 還原 typed record。歷史 cfg 不相容時發出 `CfgUnavailableWarning` 並保留有效資料，缺 cfg envelope 或壞資料仍拒絕。只有 Labber 的舊檔須先用既有 `migrations/` 工具離線轉換，不提供 fallback。核心不保存上次結果；NotebookExperiment 管理各實驗的便利引用，NotebookAdapter 只綁定可重用環境。
- `native_persistence.py` 的 `prepare_run_data` 將明確 RunRecord/spec 映射為 `PreparedRunData`，提供同源 cfg 與 SI payload 給 native、Labber writer，並帶出 grouped spec 的 Labber layout。Preparation 不做 I/O，保留 raw comment、single/grouped representation，驗證 typed cfg/result、schema 與 shape，不增加 Labber common-grid 限制。`PersistableExperiment.persistence_spec()` 公開 instance 的保存宣告，缺宣告明確失敗。`load_run_with_metadata` 讀回 typed record 與完整歷史 RunMetadata，延續 run 身分、時間、completion、context、provenance 與 device snapshot；舊 `load_run` 仍回 record／snapshot tuple。
- `native_persistence.py` 的 `native_schemas` 由同一宣告投影出 datafile disk-unit schemas，供離線 caller 使用，不要求 caller 重組 axes／variable mapping。`save_run`／`load_run` 與 instance 同名入口使用顯式 AxesSpec／GroupedAxesSpec。框架把 Result 映射成 SI payload，再由 datafile 驗證與讀寫 native HDF5。公開 `restore_run(stored, spec=...)` 不做 I/O，先核對 tag 與宣告的 variables、axes、units、dtype、shape，再還原 memory units 與 typed cfg。`PersistableExperiment.from_stored` 與 native load wrappers 共用這個轉換。合法 cfg envelope 的 cfg type、schema major 或 concrete validation 不相容時保留資料，回傳 cfg=None，並發出 `CfgUnavailableWarning`。具名 `RestoredRun` 同時提供 record 與同一原因的 cfg_issue；它以 record 型別為泛型，能保留 core typed record 或 adapter opaque result 的型別資訊。缺 /cfg 或非法 envelope 仍拒絕。Unknown cfg keys 留在 generic StoredRun，不塞進 typed cfg；typed validation 不改 raw cfg 或歷史 metadata。明示的本機單檔 `save` 保留 Labber 匯出，`load` 與 GUI Load Data 共用 native restore；兩者不宣稱格式對稱。GUI Load Data 由 LoadService 讀成 StoredRun，再交 adapter.restore／核心 from_stored；GUI 持有 opaque record 與 cfg_issue，bound-point 保存使用共用 Artifact saver。
- `artifacts` 的 `save_artifacts` 編排一次保存。正規資料寫入 `Database/<name>/runs/<experiment>_<run_id>/data.h5`，run_id 保留完整值。Run 圖片與窗格 JSON 共用此目錄；Result 的 point／全域圖片副本為 `<experiment>_<run_id>_<圖名>.png`。實驗名中的 `/` 轉 `_`。`SaveLayout` 只計算路徑，writer 在首次 Save 建立所需 Database 目錄，entry create／open 不建立它。Labber 維持日期樹內的 `<experiment>_<id6>_<序號>.hdf5`，每次另產新檔。同 run 重存覆寫 native、PNG 與窗格 JSON，不覆寫 Labber。

  固定 DataItem 共用同一 PreparedRunData，FigureItem 交原生 Figure，AnalysisItem 交 AnalysisDocument。框架每項編碼一次，對等路徑共用 bytes。DataItem.from_run 將 typed record 的準備錯誤保留為該項目的失敗，不假造 cfg；GUI producer hook 的準備錯誤也交給同一保存流程。輸出與編碼失敗各自隔離，每次呼叫回傳最新 SaveReport 並追加 saved 事件。Ledger 失敗直接傳播，已寫檔案保留，不重跑整筆保存或 rollback。原子替換只對 Windows 短暫 sharing／lock violation 重試約 0.5 秒。共用核心不依賴 GUI 或 Qt，由 artifacts-core-no-ui import contract 約束。`PersistableRecordExperiment` 結合 record core 與 persistence_spec，供正式 Notebook caller 保留 cfg／Result 配對。
- `acquisition.py` 集中 adapter 的 `AcquisitionEvidence`、UTC run identity 與兩種 capture。
  `capture_software_provenance` 只讀 package／Git／host 與已有的 SoC cached config，不查儀器。
  `capture_device_snapshot(devices)` 由 Run owner 在下述 checkpoint 呼叫，讀取全部 connected drivers 的設定並轉成隔離 JSON。
  量測時間和 completion 由前端 adapter 在 acquire 邊界記錄，不由 core 或 datafile writer 擷取。Evidence 複製版本 mapping，獨立於可修改的 RunRecord cfg／Result。
- Device snapshot 由兩個 Run 建立路徑各捕捉一次。GUI 在 hardware lease 後、adapter.run 前讀取。Notebook 同步執行，資源由 caller 擁有，不新增 lease；它在建立 RunContext 後、core.run 前讀取。兩路對全部已連線 device 各讀一次 get_info，只讀不寫。任一讀取失敗拒絕 Run，不進 adapter/core，不以 cached info 補值；無 device 記 `{}`。Snapshot 隨當次 Run 保存，後續 sweep 或 Save 不重讀、不覆寫。
- `context.py` 的 `RunContext` 提供單次 run 的 soc、soccfg、具名 devices、plots 與 cancel_signal。名稱到 driver 的對應在建立時固定，driver 仍由 caller 管理。RunContext 不提供儀器 setters；core 以 `context.devices[cfg.dev[role]]` 取得 driver，再使用實驗層 helper。每次 run 建立新 context、plots 與 `stop_signal.py` 的 StopSignal；硬體 handle 可重用，不把上一輪停止或錯誤狀態帶入下一輪。核心不管理前端 last_result 或呈現資源生命週期。
- `axes_spec.py` 用 `AxesSpec`、`Axis`、`ZSpec` 描述 Result 欄位如何對應檔案中的軸與資料 channel。`AxesSpec` 在宣告時核對 Result 的資料欄位，不要求 Result 包含 cfg。軸以 inner-first 排列；保存時套用宣告的單位比例，載入時還原成 Result 欄位。Native cfg snapshot 載入時由宣告的 cfg 型別驗證；Labber 匯出另以 comment envelope 保存 cfg。`PersistableExperiment.save` 使用 caller 指定的路徑，不替 caller 選檔名。對包含多個 Data Variable 的單一 Result，`GroupedAxesSpec`／`VariableSpec` 描述 variable 與欄位的映射，由 typed builder 重建純 Result；`GroupedAxesSpec.save(source, destination)` 接明確 RunRecord／Path，cfg 缺失時拒絕匯出。`labber_layout` 預設 common，要求共同 grid；明確宣告 independent 時，各 variable 保留自己的 1D／2D axes。單檔 save 與 artifact 保存都遵守這項宣告，不自動切換，也不改 native 格式。Grouped native load 由 `PersistableExperiment.load` 與 single 共用 restore，不從 Labber reconstruction 讀回。通用檔案格式與讀寫由 [datafile](../datafile/README.md) 負責；保存權責另見 [ADR-0063](../../../docs/adr/0063-persistence-ownership.md)。

## Cfg 與父層支援

- `cfg_model.py` 的 `ExpCfgModel` 繼承 package-root `zcu_tools.cfg_model.ConfigBase`，不宣告 `dev`。
  需要 device 的實驗 cfg 自行宣告角色到 device 名稱的 mapping；只有確實可選的實驗才宣告 optional dev。Dev 不保存 DeviceInfo、設定 patch 或 snapshot。
  使用 flux 的實驗 cfg 必填 `flux_unit`，值為 A、V 或 native，core 不提供隱含 default。
  Run 不預查 cfg.dev 名稱是否存在。Core 實際取 `context.devices[cfg.dev[role]]` 時，缺名即報錯並指名 device，不設定未使用的 device。Cfg 格式與必填 unit 的驗證仍保留。
  `ConfigBase` 提供欄位驗證、`with_updates` 和 `to_dict`；native restore 集中處理歷史 cfg 不相容的 warning 與 cfg=None。
- `cfg_assembler.py` 提供 `ExperimentScalar` 與 `ExperimentCfgValue`，描述 Run 接縫的 mapping 值型別。GUI Form 在 accept 前完成 module materialization 與 sweep 轉換；BaseAdapter 使用 Form 的 core cfg 型別驗證 accept 的 dump。Notebook caller 直接提供 typed cfg。
- `config.py` 提供繪圖尺寸 `figsize`；cfg model 與組裝由各自 owner 負責。
- `utils/` 提供 cfg 到 datafile comment codec 的 mapping 與單一 sweep 格式整理。Comment envelope 的 JSON 編解碼與 Labber reader 的 labels／units／shape／numeric container 驗證由 datafile 擁有；experiment 保留 typed cfg validation、memory units 與 generated coordinate 語意。儀器角色與單位政策由使用者套件的 helpers 擁有。Sweep helper 使用 program/v2 的 `SweepCfg`。`experiment.utils.make_sweep` 是 core 與 Notebook 共用的 sweep 建構入口。它與 [v2/utils](v2/README.md#v2-工具與實驗輔助) 的硬體量化、SNR 和 tracker 工具分開。

## Device 操作邊界

DeviceInfo 描述設定與讀回結果。Device driver 提供儀器本身的操作，flux 或 RF 角色由使用者套件定義。使用者套件選 driver、明示單位與操作用途；具體 helpers 位於 `zcu_lab/v2/_support/device.py`，見 [lab v2 說明](../../../zcu_lab/v2/README.md#實驗角色與-device-操作)。

Workflow 的同步 `get_device_cfg(name)` 與完整 cfg `setup_device(name, cfg)` 由宿主提供。使用者工具組合讀取、單位檢查與欄位修改。Framework 轉交完整 cfg；部分欄位的合併與 value／unit 處理由使用者工具負責。Measure MCP 的 partial `device_setup` 是另一個有界工具，見 ADR-0068。

## 子模組入口

Package root 保留公開 exports，只有存取對應名稱時才載入其 owner module；直接使用 persistence exports 仍會載入其所需依賴。

GUI 的 Library、Inspect 與 module writeback 共用 `gui.form` 與 `gui.form_adapter`，直接提交持久 Spec，不經 experiment 的第二套編輯樹。ModuleWriteback 的 proposal 是 ModuleSpec；experiment 仍只接收執行用 typed cfg。跨宿主契約見 ADR-0065、ADR-0008。

- [v2](v2/README.md)：program/v2 的通用 runtime 與量化、SNR、tracker 工具。[runtime](v2/runtime/README.md) 說明 Schedule、buffer、ProgramBuilder 與 executor 的執行機制。具體實驗與前端附件由使用者套件提供，組合根注入 catalog，不由框架反向載入。
