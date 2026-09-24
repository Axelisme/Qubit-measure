# ADR-0057：SampleTable v2 以平鋪 coordinate schema 定義跨模組 persistence contract

**狀態：** accepted

## Context

persisted SampleTable 是多個模組的共享介面：autofluxdep producer 寫入、
T1/T2 與 SampleMerge 消費、fluxdep visualizer 與 notebook `single_qubit` 讀取。
歷史 CSV 以 `calibrated mA` / `calibrated A` / `Flux` / `flux_bias` 等欄名攜帶
device value 與 flux，欄名本身不承載單位；`sample-current-unit-audit.md` 證明同名
欄位在不同 result directory 混用 A 與 mA（13 個非空 table 中 9 個實為 A、4 個實為
mA），magnitude 也無法可靠分辨（A/mA domains 可重疊）。generic `SampleTable`
是 schema-free storage module，不該承載跨模組的量測語意。

因此需要一個 flat、自描述、可機械驗證的 persisted coordinate contract，讓 producer
只寫 valid v2、consumer 只讀 v2、legacy 資料只經 explicit migration seam 進入，如同
[[0027]] 對 legacy experiment artifacts 的原則。

## Decision

1. **固定平鋪 coordinate prefix**。v2 table 的前置 coordinate 欄固定為
   `flux`、`dev_value`、`dev_unit`、`flux_int`、`flux_period`（
   `lib/zcu_tools/meta_tool/sample_schema.py` 的 `FLUX_COLUMN` /
   `DEV_VALUE_COLUMN` / `DEV_UNIT_COLUMN` / `FLUX_INT_COLUMN` /
   `FLUX_PERIOD_COLUMN` / `SAMPLE_COORDINATE_COLUMNS`），其後接 caller 的
   measurement columns。coordinate 語意完全由這五欄定義，不依賴外部 context。

2. **必填 A/V base unit，不做 inference**。`dev_value` 與 `dev_unit` 必填；
   `dev_value` 是 SI base unit 數值，`dev_unit` 只接受 literal `A` 或 `V`
   （`DeviceValueUnit`）。不依欄名、magnitude 或 f01 推測單位。

3. **optional all-or-none row frame**。`flux` 是 optional dimensionless unfolded
   flux；`flux_int` / `flux_period` 同時出現才構成 row frame，且逐列同時 finite 或
   同時 null、`flux_period > 0`。row frame 的 `dev_unit` 就是該列自己的 `dev_unit`
   （row-local）。變換以 `SampleFluxFrame` 為權威：
   `flux_from_dev_value(value) = (value - flux_int) / flux_period`，逆變換
   `dev_value_from_flux(flux) = flux * flux_period + flux_int`。

4. **explicit > row-frame > fallback 的 provenance precedence**。`resolve_sample_flux`
   依此順序解析每列的 unfolded flux，並回傳 closed per-row `sources` 與 derived
   `explicit_mask`（`explicit_mask` 只由 `sources == "explicit"` 導出，不做其他
   推測）；fallback frame 只適用於 `dev_unit` 與 fallback frame 相同的列；無法解析的
   列列名並 fail-fast。

5. **單一 validation gate**。`validate_sample_table_v2(samples, *, allow_empty=False)`
   是 v2 的唯一入口驗證：duplicate columns、legacy alias 欄、required columns、
   orphan frame columns、non-numeric / non-finite、`dev_unit` 非 A/V、non-positive
   period 全部 fast-fail；`allow_empty` 只允許完全空 table 或 zero-row valid v2
   headers。`SampleTableV2Error` 以 `reason` / `data` 承載 rejection diagnostics。

6. **Legacy 只經 explicit migration 進入，且 pure**。
   `migrate_sample_table_v2(samples, *, dev_value_column, dev_value_unit,
   flux_column=None, flux_frame=None)` 是唯一 legacy seam：A→A、V→V 原值通過，
   mA/mV ÷1000 轉 base unit；單次 invocation 單一 scalar unit；
   `LegacySampleFluxFrame` 先正規化到 base unit，且其 physical kind 必須與
   `dev_value_unit` 相符；`flux_column` 只在 caller 宣告時處理。輸入含 v2 columns、
   缺欄、`source == dest`、dest 已存在或 validation 失敗 → fail。函式不 mutate
   輸入，產出完整 target v2 table（coordinate columns 在前，measurement columns
   與 row order 原樣保留）。

7. **Operator-owned data migration**。`script/migrate_sample_table_v2.py` 以
   `SOURCE DEST --dev-value-column --dev-value-unit [--flux-column]
   [--flux-int --flux-period --frame-unit]` 操作；預設 dry-run，`--write` 只寫入
   distinct dest，temp-file + no-clobber。工具不自動掃描或改寫任何既有 CSV；
   user-owned 資料的實際改寫由 operator 以 file-scoped mutation authority 執行。

8. **generic `SampleTable` 維持 schema-free**。v2 policy 住在
   `meta_tool/sample_schema.py`，不進入 reusable storage module。

9. **生產路徑只接受 v2**。autofluxdep export（`gui/app/autofluxdep/services/
   sample_table_export.py`，run-creation 時把裝置 unit snapshot 進 manifest
   `workflow.flux.unit`，export 用 snapshot 而非 live state，unit 不確定則不 export
   v2、fail before CSV mutation）、notebook `single_qubit` producer（append 前
   validate；`dev_value = cur_value`、`dev_unit = "A"`；frame 只在 `flx_int` /
   `flx_period` 皆 finite 時寫；explicit `flux` 只給 point-level calibrated 行）、
   T1/T2（`t1_curve/workflow.py` / `t2_curve/workflow.py` 以 `resolve_sample_flux`
   取代 (1.0, 1000.0) 猜測）、SampleMerge（`fit_tools/sample_merge.py` 的
   `FluxFrame` / `SampleSource` / `merge_sample_sources`）與 fluxdep visualizer
   都只消費 v2。legacy alias（`calibrated mA` / `calibrated A` / `Flux` /
   `flux_bias`）只存在於 migration、rejection diagnostics、tests 與說明 docs。

10. **修正路徑顯式且 bounded**。`fit_tools/flux.py` 的 `correct_flux_from_f01`
    接受 GHz 輸入，只對 derived/fallback rows 做 bounded 修正（
    `max_abs_flux_correction` 上限），explicit flux rows 永不自動修正；
    `align_flux_to_window` 以 per-flux integer branch（`k ∈ ℤ`）對齊，window 必須
    finite、strictly increasing 且 width ≤ 1.0，width 超過一個 period 直接 fail。
    分析 bias 不寫回 persistence。

## Consequences

- 任何 v2 table 單看檔案即可決定語意：單位、frame、provenance 全部自描述，unit
  不依賴外部 context。
- Producer 與 consumer 解耦：producer 只負責寫 valid v2，consumer 只負責解析，
  解析失敗 fast-fail；沒有 runtime dual-schema compatibility layer。
- Legacy 資料必須經 operator 執行 migration script 才能被 v2 consumers 使用；
  migration 的輸出是完整 v2，輸入不被修改。
- 修正與對齊都顯式、bounded、且只作用於 fitted-derived rows，不把分析假設固化進
  persisted 資料。

## Rejected alternatives

- **canonical mA/A single numeric column**：audit 證明同名歷史資料已混用，且
  mA/mV 把 display multiplier 帶進 persistence。
- **`flux_bias` column**：現有 predictor 已用此名稱表示 device-unit affine offset；
  point correction 以 explicit `flux` 與 nominal derived flux 的差表達即可。
- **每列重複完整 provenance / frame object**：不適合 flat CSV；v2 只保存 row 所需
  coordinate fields。
- **省略 `dev_unit`**：standalone CSV 會再次無法區分 current 與 voltage，也無法
  驗證 frame physical kind。
- **依欄名、magnitude 或 f01 自動判定 legacy unit**：A/mA domains 可重疊，audit
  也證明同 directory 混用。
- **runtime 永久接受 v1/v2 dual schema**：會把 migration ambiguity 變成長期
  compatibility layer；legacy 只經 explicit migration seam 進入。
- **把 v2 policy 硬編進 generic `SampleTable`**：破壞 reusable storage module。
