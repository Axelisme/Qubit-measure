# `zcu_tools.resources.sample_table` — sample records

**Last updated:** 2026-09-27 — resource owner split

## `SampleTable`（`table.py`）

以 CSV 儲存量測樣品紀錄（pandas DataFrame 包裝）。`SampleTable` 本身不固定 schema；notebook 與 GUI/script export 以呼叫端約定欄位，例如 single-qubit producer 使用 v2 的 `dev_value`／`dev_unit`，再加 `Freq (MHz)`、T1/T2r/T2e、error、`comment` 與 `date` 欄位。舊 CSV 的 `calibrated mA` 只供明示 legacy conversion，不是新樣品的欄位。`extend_samples()` 會接在既有 CSV 後面；要重建檔案需由 caller 先刪除或使用 export script 的 overwrite mode。

```python
st = SampleTable("samples.csv")  # 也接受 pathlib.Path
st.add_sample(dev_value=1.23e-3, dev_unit="A", flux=0.25, T1=50e-6)
df = st.get_samples()
```

**方法**：`add_sample(**kwargs)`、`extend_samples(**kwargs)`（批次）、`update_sample(idx, **kwargs)`、`get_samples()`。

## SampleTable v2 協定（`schema.py`）

v2 是**深層 opt-in** 的 flat coordinate 契約：generic `SampleTable` 維持 schema-free，既有 CSV 的 custody 絕不自動變更；只有明確呼叫 contract API（`validate_sample_table_v2` / `resolve_sample_flux`）或明確呼叫純轉換函式的資料才受 v2 欄位規則約束。指定 runtime producer/consumer 顯式呼叫並強制 v2 contract：AutoFlux exporter（`gui/app/autofluxdep/services/sample_table_export.py`）、notebook `single_qubit` producer（`notebook_md/single_qubit.md`）、T1/T2（`notebook/analysis/t1_curve/workflow.py` / `notebook/analysis/t2_curve/workflow.py`）、SampleMerge（`notebook/analysis/fit_tools/sample_merge.py`）與 fluxdep/design（`notebook/analysis/fluxdep/utils.py` / `notebook/analysis/design/search.py`）。v2 欄位全為英文小寫 snake_case：`flux`、`dev_value`、`dev_unit`、`flux_int`、`flux_period`（`SAMPLE_COORDINATE_COLUMNS`）。

**驗證（`validate_sample_table_v2`）**：`dev_value`/`dev_unit` 必填，unit 僅限 A/V；coordinate 只接受 real numeric（拒絕 datetime/timedelta、boolean、complex），required value 必須 finite；`flux` 可空、只接受有限值或 null；`flux_int`/`flux_period` 成對出現，每 row 兩者同為有限或同為 null，period 必須 > 0；拒絕 legacy 別名欄位（`calibrated mA`、`calibrated A`、`Flux`、`flux_bias`）、重複欄位與孤立 frame 欄位。

**flux 解析（`resolve_sample_flux`）**：closed per-row 來源順序 `explicit → row-frame → fallback-frame`，`sources` 為每 row 的權威來源、`explicit_mask` 只由 `sources` 推導；任一 row 無法解析即 fail-fast 列出 index。fallback frame 的單位必須與該 row 的 `dev_unit` 一致。

**legacy 遷移（`migrate_sample_table_v2`）**：caller 明確提供單一 A/mA/V/mV 來源單位，函式將 dev value 與宣告的 frame 縮放到 base A/V，回傳經驗證的 v2 DataFrame，不推論、不 alias，也不改寫輸入。repo 不提供 CSV 遷移 CLI；若 operator 要儲存結果，須自行核對來源欄位與 flux/frame 證據，取得目的檔寫入授權，並以不覆蓋來源或既有目的檔的方式輸出。
