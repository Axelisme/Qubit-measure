# `qubit_params` 參考

**Last updated:** 2026-09-27 — moved from module docstring

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

**未知 section preservation**：typed 寫入只更新自己的 section，其它未知 section 會保留。`to_raw()` / `replace_raw()` / `update_raw()` 只供 `notebook.persistance` 舊 helper 過渡使用；新 caller 應使用 typed 方法。

**Active noise params**：`t1_curve_fit` 的 `fixed`、`free`、`bounds`、`init` 與 `stderr` 只能提到 active params。
