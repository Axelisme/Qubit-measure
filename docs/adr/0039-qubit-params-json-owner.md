---
status: accepted
---

# ADR-0039 — QubitParams owns params.json typed handoff

關聯 [[0029]]（prediction engine 擁有 simulation policy）、[[0037]]（value lookup 不取代 explicit typed dependencies）。

## 脈絡

`result/<chip>/<qub>/params.json` 是多個 workflow 的 handoff file：

- measure-gui / setup 需要 project identity 與 result scope discovery。
- fluxdep-gui 寫入 `fluxdep_fit`（EJ/EC/EL + flux alignment）。
- dispersive-gui 讀 `fluxdep_fit` 並寫入 `dispersive`（g, bare_rf）。
- T1 curve notebook / analysis 寫入 `t1_curve_fit`（noise params + fit metadata）。
- `FluxoniumPredictor.from_file()` 與 predictor dialog 從同一份 file 讀出 fluxonium model。
- notebook helper `dump_result` / `load_result` / `update_result` 是舊 import path。

這些 caller 原本各自知道 JSON section shape、missing-section error、project migration 與保留/覆蓋規則。這讓檔案格式改動需要掃多個 app/service，也讓 section-level metadata（例如最後修改時間）容易不一致。

## 決策

`meta_tool.QubitParams` 是 `params.json` 的唯一 typed 讀寫 module。

- `QubitParams` 繼承 `SyncFile`，沿用 repo 既有 mtime sync 模型。
- Known sections 透過 typed value objects 進出：`ParamsProject`、`FluxDepFit`、`DispersiveFit`、`T1CurveFit`、`FluxoniumModelParams`、`DispersiveFitInputs`。
- Unknown sections 在 typed write 時保留；這讓新 section 可與舊 caller 共存。
- `fluxdep_fit`、`dispersive` 與 `t1_curve_fit` 視為獨立 module section；寫入其中一個 section 不會自動刪除另一個 section。
- `set_fluxdep_fit(...)`、`set_dispersive_fit(...)` 與 `set_t1_curve_fit(...)` 都會在自己的 section 寫入 `timestamp`，記錄最後修改時間。
- `set_dispersive_fit(...)` 要求檔案已存在且已有 `fluxdep_fit`，避免建立沒有 upstream handoff 的半成品。
- `set_t1_curve_fit(...)` 要求檔案已存在且已有 `fluxdep_fit`，並只保存後續模擬 handoff 需要的 fit params 與 metadata，不把 sample arrays 或 dense model curves 放進 `params.json`。
- `t1_curve_fit.params` 使用 white-list noise channel 語義：`Temp` 必填，`Q_cap` / `x_qp` / `Q_ind` 只在該 channel 納入 all-in-one fit 時出現；`fixed`、`free`、`bounds`、`init` 與 `stderr` 只能提到 active params。
- Project identity migration 屬於 `QubitParams`：canonical `project.{chip_name, qubit_name}` 優先；缺 canonical project 時，result-scope discovery 才用 `result/` 下路徑推導 identity。
- `gui.result_scope` 只負責掃描 / path derivation / scope mismatch policy；它委派 `QubitParams` 做 `params.json` migration。
- `FluxoniumPredictor.from_file()` 與 predictor dialog 不解析 JSON；它們只要求 `QubitParams.require_fluxonium_model(...)`。
- `notebook.persistance` 的 `dump_result` / `load_result` / `update_result` 保留為 transitional wrapper，內部委派 `QubitParams`，不再擁有另一套 JSON implementation。

## 拒絕的替代方案

- **`dataclass QubitParams` + `save/load` helper**：它只把 JSON shape 包成資料結構，仍會要求 caller 自己決定 section-level merge、project migration、section timestamp、missing-section error 與 unknown-section preservation。刪掉 helper 後，複雜度會回到 fluxdep/dispersive/predictor caller，module 深度不足。
- **讓 `MetaDict` 管理 params.json**：`MetaDict` 是任意 metadata store；`params.json` 是跨 workflow typed handoff file，有 section 不變式與 migration policy。把兩者混在一起會讓 caller 仍然知道 raw key。
- **把 policy 放在 GUI service**：會重複出現在 fluxdep、dispersive、measure predictor 與 result-scope discovery；違反單一讀寫權威。

## 後果

- `params.json` schema knowledge 集中在 `meta_tool`。
- GUI / predictor caller 以 typed methods 互動，不再知道 raw JSON nesting。
- fluxdep、dispersive 與 T1 curve fit 結果各自保留；caller 可用 section `timestamp` 判斷相對新舊。
- 舊 notebook import path 仍可用，但新 code 不應新增直接使用 raw helper 的 production path。
