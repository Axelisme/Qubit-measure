# zcu_tools.datafile

**Last updated:** 2026-10-06，Cfg snapshot validation


`zcu_tools.datafile` 是 Labber 與 native experiment data file 的 public
facade。caller 優先從 package root import model 與 function。

- `save_run_data`／`load_run_data` 處理版本化 `zcu.experiment-data` HDF5。
  `ExperimentPayload` 使用 SI／離散單位與明示 single／grouped representation，
  不同 variables 可以使用不同網格。`VariableSchema` 與
  `validate_experiment_payload` 集中檢查 disk labels、units、dtype 與 shape。
- `validate_cfg_snapshot` 擁有 cfg 的 JSON、非空 cfg_type 與完整 major.minor 格式規則。
  Native writer/reader 與離線 migration callback 共用它；純驗證不修改歷史值。
  Concrete cfg 的欄位及支援 major 由該 cfg owner 驗證，不屬於檔案格式。
- `RunMetadata`、`RunSnapshot`、`CfgSnapshot` 是 caller 傳入的歷史資料，writer
  不擷取 live context。Native saving 使用 caller 的 exact path，先寫同目錄 temp，
  關檔後發布；`replace=True` 原子替換，失敗清理 temp 並保留原 destination。
- Native loading 預設只投影已知內容。明示 `preserve_unknown=True` 才取得
  `NativeExtensions` detached HDF5 image。Generic caller 把 image 傳回 writer，
  保留同 major minor 的未知 JSON／HDF5 內容與不改形 rewrite 的 references。
  Typed experiment tuple 不是 lossless rewrite envelope，也沒有 legacy fallback。

- `write_labber` 從同一 `ExperimentPayload` 輸出 single 或 canonical
  common-grid grouped Labber，包括單成員 grouped。它只寫 caller 的 exact path，
  不加副檔名、不覆寫、不呼叫 native writer。兩個格式的失敗隔離由 caller 編排，
  batch-level 保存流程不在 datafile。Labber 至少需要一個 step axis，coordinates
  必須可表示成 real；scalar 在建立目的檔前拒絕，不影響 native 的 scalar 契約。
- `encode_labber_comment`／`decode_labber_comment` 與 `LabberComment`
  擁有 cfg/comment/timestamp envelope。不屬於 envelope 的手寫文字原樣保留；
  可辨識 envelope 的欄位型別錯誤直接報錯，不猜歷史 cfg 或量測時間。
- `validate_labber_payload`／`cast_labber_values` 擁有 Labber reader 的
  label、unit、shape 與 numeric real/complex container 契約。
  Experiment 只宣告 schema、換 memory units，並驗證 generated coordinate 語意。
- `Axis`、`LabberPayload`、`LabberMetadata`、`LabberData` 描述單一
  inner-first axes 的 Labber dataset。
- `DataVariable`、`GroupedLabberData` 描述 grouped experiment dataset：單一
  experiment data file 內含多個 variable payload，metadata 共用。
- `load_legacy_labber_payload` 是離線 migration 專用入口。Caller 提供
  明確 disk schema；它使用 single 或 marker-qualified grouped reader，驗證
  labels／units／shape 並轉換 numeric containers，不猜 variable identity 或單位。
  正常 runtime loader 不呼叫這個入口。
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
