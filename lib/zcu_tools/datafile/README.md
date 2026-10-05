# zcu_tools.datafile

**Last updated:** 2026-10-05，Data Variable 命名


`zcu_tools.datafile` 是 Labber-style experiment data file 的 public
facade。caller 優先從 package root import model 與 function。

- `Axis`、`LabberPayload`、`LabberMetadata`、`LabberData` 描述單一
  inner-first axes 的 Labber dataset。
- `DataVariable`、`GroupedLabberData` 描述 grouped experiment dataset：單一
  experiment data file 內含多個 variable payload，metadata 共用。
- `save_labber_data` / `load_labber_data` 處理 single-variable file。
- `save_grouped_labber_data` / `load_grouped_labber_data` 處理 canonical one-shot
  grouped v2。所有 variables 必須共享完全相同的 inner-first axes、shape 與
  timestamps。saver 在建立目的檔前驗證 common-grid contract，並把 variables 寫成
  root Labber log 的平行 scalar channels。ordered variable-to-channel attrs 保存 domain
  identity；experiment loader 傳 required variables，省略 required variables 只用於
  diagnostic inspection。
- unmarked grouped v1 不屬於 runtime 可載入格式；loader 明確要求 canonical
  grouped v2，不改寫 input。marker-qualified streaming grouped v1 仍走獨立 decoder。
- `StreamingLabberVariableSpec` / `open_streaming_grouped_labber_data` 處理
  grouped Labber file 的 partial-write use case。它保留 marker-qualified grouped v1
  root/`Log_N` layout 與 heterogeneous axes，不共用 one-shot v2 decoder。
  `open_streaming_labber_data`
  是 single-log `save_labber_data` 的 streaming 對偶。caller 先宣告 full-shape
  schema，writer 預建 nan-filled datasets，之後以 outer row slice 寫入並
  flush。它是長掃 workflow 的 streaming primitive，不改變 one-shot save helper
  的 complete-file 語意。
- Streaming writer 對自己建立的 Labber `Data` layout 採 hard invariant：若
  內部 HDF5 group / dataset 結構不符合預期，立即 raise，不做靜默 fallback。
- Experiment semantic schema 住在 `zcu_tools.experiment.axes_spec`：
  `GroupedAxesSpec` / `VariableSpec` 把 Result/Cfg 映射到這裡的 generic grouped
  payload；`datafile` 不反向依賴 experiment Result 或 cfg。
- Save helpers 寫入 caller 傳入的 formatted path；既有目的地 fast-fail，不自動
  suffix 或覆寫。
- Path helpers（`format_ext`、`reserve_labber_filepath`、datafolder helpers）由
  facade re-export；`reserve_labber_filepath` 只供 caller / orchestration layer
  預先決定 unique final path。
- `format_ext` / `remove_ext` 只處理檔名 suffix，不改路徑中段的 `.h5` /
  `.hdf5` 子串；`reserve_labber_filepath` 保留 Labber-style numeric sequence，
  但孤立的數字尾碼可作為 caller 命名的一部分。

`datafile/` 內部 module 是責任拆分，不是額外 public import path。

datafile 只負責資料檔格式、讀寫及 streaming；Result 與 axes mapping 屬於
`zcu_tools.experiment.axes_spec` 所在的 experiment 層，artifact policy 屬
workflow，路徑意圖由 caller 決定。
