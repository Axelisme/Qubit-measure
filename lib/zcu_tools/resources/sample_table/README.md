# `zcu_tools.resources.sample_table` — sample records

**Last updated:** 2026-10-11，JSONL optional producer fields and accepted timestamps

## 條目樣本（`jsonl.py`）

`ResultEntry.samples(component)` 回傳 `JsonlSampleTable`，綁定 `results/<name>/samples/<component>.jsonl`。JSONL 是只追加的樣本歷史；`read()` 回傳每個 `run_id` 最後一整行的 DataFrame，不合併舊行欄位。不存在的檔案回空表，損壞行則回報檔名與行號，不跳過。

`append` 要求 `run_id`。`dev_value`、`dev_unit` 與其他欄位皆由 producer 選填，保留原值與 caller 單位，不檢查 dev 存在或換算單位。`point` 可為空，`at` 預設由儲存層填 UTC，也可明示 accepted 事件的 UTC 時間。`source` 預設 manual，也可由 caller 提供 accepted 事件 ID 或 `workflow:<run_id>`。Producer 欄位可以保存巢狀 JSON。儲存層不計算 flux 座標，不讀 ledger 或元件 schema；下游查詢與 v2 驗證自行處理缺值。

`export_csv(path, overwrite=False)` 只匯出最新列，不輸出 pandas index；既有目的檔預設拒絕覆寫。自身來源 JSONL 永遠拒絕作為目的地，包含 symlink 或 hardlink，不因 overwrite 放行；只保護自身 source，不掃描其他檔案。JSONL 與 CSV 匯出不提供跨檔交易、多 writer 協調或 crash recovery。舊 CSV API 保留給尚未遷移的 caller，不作 JSONL 的 fallback。

## 舊 CSV `SampleTable`（`table.py`）

`SampleTable` 是保留給尚未遷移 caller 的 CSV API，以 pandas DataFrame 儲存量測樣品紀錄。它不固定 schema，欄位由 caller 約定，與 JSONL 讀取 API 分開。`extend_samples()` 追加到既有 CSV；重建檔案由 caller 明確安排。

```python
st = SampleTable("samples.csv")  # 也接受 pathlib.Path
st.add_sample(dev_value=1.23e-3, dev_unit="A", flux=0.25, T1=50e-6)
df = st.get_samples()
```

**方法**：`add_sample(**kwargs)`、`extend_samples(**kwargs)`（批次）、`update_sample(idx, **kwargs)`、`get_samples()`。

## SampleTable v2 協定（`schema.py`）

v2 定義 flat coordinate 契約。Generic `SampleTable` 不固定 schema，也不自動轉換既有 CSV。呼叫端明確使用 `validate_sample_table_v2`、`resolve_sample_flux` 或純轉換函式時，才套用座標規則。Notebook 的 T1/T2 workflow、SampleMerge、fluxdep utils 與 design search 使用此契約。座標欄位為 `flux`、`dev_value`、`dev_unit`、`flux_int`、`flux_period`（`SAMPLE_COORDINATE_COLUMNS`）。

**驗證（`validate_sample_table_v2`）**：`dev_value`/`dev_unit` 必填，unit 為 A/V/native；coordinate 只接受 real numeric（拒絕 datetime/timedelta、boolean、complex），required value 必須 finite；`flux` 可空、只接受有限值或 null；`flux_int`/`flux_period` 成對出現，每 row 兩者同為有限或同為 null，period 必須 > 0；拒絕 legacy 別名欄位（`calibrated mA`、`calibrated A`、`Flux`、`flux_bias`）、重複欄位與孤立 frame 欄位。

**flux 解析（`resolve_sample_flux`）**：closed per-row 來源順序 `explicit → row-frame → fallback-frame`，`sources` 為每 row 的權威來源、`explicit_mask` 只由 `sources` 推導；任一 row 無法解析即 fail-fast 列出 index。fallback frame 的單位必須與該 row 的 `dev_unit` 一致。native 不隱含 A/V 換算；caller 的 frame 明示 native device value 到 flux 的映射。JSONL 寫入不計算 flux，notebook consumers 以 `JsonlSampleTable.read()` 取得最新 rows 後才解析座標。

**legacy 遷移（`migrate_sample_table_v2`）**：caller 明確提供單一 A/mA/V/mV 來源單位，函式將 dev value 與宣告的 frame 縮放到 base A/V，回傳經驗證的 v2 DataFrame，不推論、不 alias，也不改寫輸入。離線的 [root migrations 工具](../../../../migrations/README.md) 提供公開轉換函式與 `python -m migrations` CLI。它明示改名有來源證據的舊量測欄位，重用這裡的座標轉換，並經 `ResultEntry.samples(component)` 追加 JSONL。來源檔不變；無法確定的列不寫入，列入報告。工具不操作硬體，也不自行選擇真實資料。
