# `qubit_params` 參考

**Last updated:** 2026-10-11，直接重新讀取 JSON 與原子寫回

本檔是 [`qubit_params.py`](qubit_params.py) 的 owner 參考。家族導覽見 [resources README](README.md)。

## `QubitParams`

`QubitParams` 是 `result/<chip>/<qub>/params.json` 的唯一 typed 讀寫 module。它不是任意 key/value store；caller 透過語意方法讀寫 project identity、`fluxdep_fit`、`dispersive`、`t1_curve_fit` 與 predictor 所需的 fluxonium model。

**主要責任**：

| 方法 | 說明 |
|------|------|
| `ensure_project(project)` | 建立或更新 canonical `project.{chip_name, qubit_name, resonator_name}`，並同步 legacy `name` |
| `migrate_project_from_path(result_root=...)` | 將 v0 identity 原地升級；canonical `project` 優先，缺 project 時才由 `result/` 路徑推導 |
| `set_fluxdep_fit(fit)` | 寫入 fluxdep fit；保留獨立的 `dispersive` section，並更新 `fluxdep_fit.timestamp` |
| `require_dispersive_inputs(default_bare_rf=...)` | 讀出 dispersive GUI 的硬輸入，並集中 `bare_rf` seed 優先序 |
| `set_dispersive_fit(fit)` | 寫入 dispersive fit；要求檔案已存在且已有 `fluxdep_fit`，並更新 `dispersive.timestamp` |
| `set_t1_curve_fit(fit)` | 寫入 T1 curve noise fit handoff；要求檔案已存在且已有 `fluxdep_fit`，並更新 `t1_curve_fit.timestamp` |
| `require_t1_curve_fit()` | 讀出 T1 curve fit 的 noise params、stderr 與 fit metadata |
| `require_fluxonium_model(flux_bias=...)` | 讀出 `FluxoniumPredictor` / sim 需要的 `(EJ, EC, EL, flux_half, flux_period, flux_bias)` |

**獨立 section 與 timestamp**：`fluxdep_fit`、`dispersive` 與 `t1_curve_fit` 是獨立 module section；寫入其中一個不會刪除另一個。每次 typed 寫入會更新該 section 的 `timestamp`，供 caller 判斷最後修改時間。`t1_curve_fit` 只保存後續模擬需要的 fit params 與 metadata；sample arrays 和 dense model curves 不放進 `params.json`。`t1_curve_fit.params` 中 `Temp` 必填，`Q_cap` / `x_qp` / `Q_ind` 只保存 active noise channel；省略某個 noise key 表示該 channel 未納入 all-in-one fit。

**未知 section preservation**：typed 寫入只更新自己的 section，其它未知 section 會保留。`to_raw()` / `replace_raw()` / `update_raw()` 保留為低階原始資料操作；caller 應使用 typed 方法。

## 同步與持久化

`QubitParams` 直接持有檔案 path 與 readonly。建構時若檔案已存在，會讀取 JSON。公開讀取和寫入前重新讀取既有檔案，不依賴 mtime、dirty flag 或 conflict warning。

寫入先完成 JSON 序列化，再把完整內容寫入同目錄的暫存檔。關閉暫存檔後原子替換目標，最後清除暫存檔。序列化或替換前的寫入失敗不會截斷既有檔案。這不是跨程序鎖或交易；並行寫入仍可能覆蓋對方的修改，不提供跨檔案或掉電保證。

readonly 在寫入方法入口拒絕寫入，拋出 `RuntimeError`，不修改資料或建立檔案。path 為 `None` 時只操作記憶體。

檔案模式的讀取要求檔案已存在，否則拋出 `FileNotFoundError`。`update_raw()`、`set_dispersive_fit()`、`set_t1_curve_fit()` 與 identity 遷移也要求檔案已存在。`ensure_project()`、`replace_raw()` 與 `set_fluxdep_fit()` 可以建立檔案和父目錄。

JSON 語法錯誤仍以 `QubitParamsError(reason_code="params_read_failed")` 回報，非 object 根節點以 `params_not_object` 回報。其它 I/O 例外與 raw 序列化的 `TypeError` 直接傳遞。

舊繼承介面 `load`、`dump`、`sync`、`update_modify_time`、`has_persistence` 與 `require_writable` 已退休。Caller 使用上表的 typed 方法或三個 raw 方法，不需要手動同步。

**Active noise params**：`t1_curve_fit` 的 `fixed`、`free`、`bounds`、`init` 與 `stderr` 只能提到 active params。
