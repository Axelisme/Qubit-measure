# `zcu_tools.experiment` — 實驗層

**Last updated:** 2026-10-01 — RunRecord 與預設 record codec

`experiment/` 定義實驗介面、Result 與保存資料之間的映射，以及實驗 cfg 的支援機制。這個位置不表示其現有程式碼均與 QICK 無關。`cfg_assembler.py` 從 `program.v2` 匯入 `ModuleCfgFactory`，`utils/sweep.py` 使用 `SweepCfg`。程式指令的建構與 acquire 由 [program/v2](../program/v2/README.md) 及 [experiment/v2/runtime](v2/runtime/README.md) 接合。

## 實驗介面與資料

- `records.py` 的 `RunRecord` 將 nullable cfg 與純 Result 配對。建立時複製 cfg，隔離 caller 之後對輸入 cfg 的修改；record 內 cfg 與 Result 仍可修改。`interfaces.py` 的 `RecordExperiment` 定義 run／save／load，`SynchronousExperiment` 另加同步 analyze，不要求 interactive 實驗提供 analyze。
- `base.py` 的 `PersistableExperiment` 透過 `AXES_SPEC` 提供無操作狀態的 `save(source, destination)`／`load(source)`，路徑使用 `Path`。保存接明確 `RunRecord`，cfg 缺失時拒絕。載入回傳 `RunRecord`，cfg 無效時 warning 並保留有效資料，不更新 `last_result`。Notebook convenience 的引用管理由 caller 負責。`ExperimentProtocol`、`AbsExperiment` 與 `record_result`／`retrieve_result` 仍服務尚未遷移的實驗，不代表新核心契約。舊 caller 的 filepath-first 保存與依賴載入快取的用法尚待逐項遷移，本模組不偵測新舊簽名。
- `context.py` 的 `QickContext` 明確提供單次 run 的 soc、soccfg 與 plots。Caller 可重用硬體 handle，但每次操作建立新 plots；核心不管理前端的 last_result 或呈現資源生命週期。
- `axes_spec.py` 用 `AxesSpec`、`Axis`、`ZSpec` 描述 Result 欄位如何對應檔案中的軸與資料 channel。`AxesSpec` 在宣告時核對 Result 的資料欄位，不要求 Result 包含 cfg。軸以 inner-first 排列；保存時套用宣告的單位比例，載入時還原成 Result 欄位。RunRecord 的 cfg 經 comment 保存，載入時由 cfg 型別驗證還原。`PersistableExperiment.save` 使用 caller 指定的路徑，不替 caller 選檔名。對包含多個 Dataset Role 的單一 Result，`GroupedAxesSpec`／`RoleSpec` 描述 role 與欄位的映射，載入後由 typed builder 重建 Result；這條 grouped 路徑仍使用 Result 的 `cfg_snapshot`，另待遷移。通用檔案格式與讀寫由 [datafile](../datafile/README.md) 負責；保存權責另見 [ADR-0063](../../../docs/adr/0063-persistence-ownership.md)。

## Cfg 與父層支援

- `cfg_model.py` 的 `ExpCfgModel` 繼承 package-root `zcu_tools.cfg_model.ConfigBase`，提供可選的 `dev` 欄位。`ConfigBase` 提供欄位驗證、`with_updates` 和 `to_dict`；`ExpCfgModel.validate_or_warn` 嘗試驗證載入的 cfg，失敗則發出警告並回傳 `None`。
- `cfg_assembler.py` 的 `assemble_experiment_cfg` 使用 caller 傳入的 module library 與 device snapshot，套用 overrides、組裝 device／module／單一 sweep 資料，再驗證成目標實驗 cfg。`make_cfg` 在未指定 snapshot 時向 `GlobalDeviceManager` 取目前資訊。這是量測 cfg 的組裝，不是 GUI 的編輯模型。
- `config.py` 目前提供繪圖尺寸 `figsize`；它不是 cfg model，也不執行 cfg 組裝。
- `utils/` 提供實驗側的 comment JSON 編解碼、device 設定／啟用與單一 sweep 格式整理。其中 device helper 會呼叫 `GlobalDeviceManager`，sweep helper 會使用 program/v2 的 `SweepCfg`。它與 [v2/utils](v2/README.md#v2-工具與實驗輔助) 的硬體量化、SNR 和 tracker 工具分開。

## 子模組入口

- [v2](v2/README.md)：使用 program/v2 的實驗家族與具體 workflow；[實驗撰寫檢查清單](v2/README.md#寫新-experiment-時的檢查清單) 說明 v2 實驗的撰寫慣例。[runtime](v2/runtime/README.md) 說明 Schedule、buffer、ProgramBuilder 與 executor 的執行機制。
- [v2_gui](v2_gui/README.md)：measure 與 autofluxdep 的 GUI 實驗接入家族，具體 GUI workflow／policy 留在 app。
- [cfg_editing](cfg_editing/README.md)：Qt-free 的實驗側 program cfg 編輯支援；與執行前 cfg 組裝分工。
