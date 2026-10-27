# `zcu_tools.resources.record_table` — component records

**Last updated:** 2026-10-27

**讀者：** framework-dev。JSONL 樣本歷史、CSV 匯出與 v2 座標協定。

## 條目樣本（`jsonl.py`）

`ResultEntry.records(component)` 回傳 `JsonlRecordTable`，綁定 `result/<name>/records/components/<component>.jsonl`。唯讀 `path` 提供該表 exact bound JSONL 路徑，供 caller 報告目的地或檢查同檔。綁定與查詢路徑只操作記憶體。JSONL 是只追加的樣本歷史；`read()` 回傳每個 `run_id` 最後一整行的 DataFrame，不合併舊行欄位。不存在的檔案回空表。讀取只檢查 JSON object 與選取最新列所需的非空字串 `run_id`；其他欄位、NaN 與時間等 metadata 原樣保留，不套寫入 schema。JSON 語法損壞、缺 run_id 或未完成的尾行回報檔名與行號，不跳過。`latest(run_id)` 共用同一讀取路徑，只回傳指定 run 的最新整行。`read(flatten=True)` 將巢狀物件鍵串成點號欄名，例如 `devices.bias.value`；list 保留為 cell 值。同一列的不同來源路徑若產生相同點號欄名，會拋出 ValueError，指出來源檔、從零起算的輸出列、run_id 與衝突欄名。沒有衝突的點號鍵照常保留。預設 `read()` 仍保留 nested cells 與原始點號鍵，不要求攤平。兩者都不改寫來源。

`append` 要求 `run_id`，成功後回傳當筆已序列化的隔離 JSON row，含 component、run_id、point、at、source 與 producer 欄位，不再讀回檔案。`component` 取自 table binding，producer 同名欄位不能覆寫它。直接使用 `JsonlRecordTable(path, component=...)` 時必須明示非空元件名，不從檔名推論；省略 component 只供讀取與匯出，append 會在寫檔前報 ValueError。舊列缺 component 仍可讀，不補值或改寫。`dev_value`、`dev_unit` 與其他欄位皆由 producer 選填，保留原值與 caller 單位，不檢查 dev 存在或換算單位。`point` 可為空，`at` 預設由儲存層填 UTC，也可明示 accepted 事件的 UTC 時間。`source` 預設 manual，也可由 caller 提供 accepted 事件 ID 或 `workflow:<run_id>`。Producer 欄位可以保存巢狀 JSON。儲存層不計算 flux 座標，不讀 ledger 或元件 schema；下游查詢與 v2 驗證自行處理缺值。

`export_csv(path, overwrite=False)` 匯出最新列的資料欄位；既有目的檔預設拒絕覆寫。自身來源 JSONL 永遠拒絕作為目的地，包含 symlink 或 hardlink，不因 overwrite 放行；只保護自身 source，不掃描其他檔案。JSONL 與 CSV 匯出不提供跨檔交易、多 writer 協調或 crash recovery。舊 CSV 的讀取與轉換由 root `migrations/` 提供，不作 JSONL 的 fallback。

## 明確的歷史升級（`upgrade.py`）

`upgrade_record_files(paths, device_name=..., apply=False)` 是舊 `dev_value/dev_unit` pair 的明確升級接縫。名稱由 operator 宣告，重用 `cfg_model.validate_device_name`，不從 setup 或路徑推測。它按實體行回傳具名 `RecordRowUpgrade` 分類與 `RecordUpgradeResult` 檔案結果，不經 `read()` 的最新列去重，也不經 append。只檢查舊 pair 與目標 named pair，保留其他 device entry、metadata、未知 payload、列順序與重複 run_id。目標 metadata-only entry 可合入 pair；任一舊欄位與目標 pair 欄位共存則拒絕，不覆寫。

Dry-run 不寫檔。Apply 在 record owner 內取得 upgrade sidecar lock、重讀、備份原 bytes，再原子替換。任一 partial/corrupt 列拒絕整檔並列行號，其他檔繼續。合法新列原 bytes 保留；已升級檔重跑跳過。一般 append 不參與這把 upgrade lock，operator 必須先停止 concurrent writers。它不合成 frame/component，不增加 record 版本，也不更改 component schema_version。完整分類、CLI、備份還原與改名規則見 [操作指南](../../../../docs/record-device-upgrades.md)。

## 裝置投影（`projection.py`）

裝置投影的 `device_name` 與 flux row-frame keys 共用 `cfg_model.validate_device_name`。
它拒絕 ASCII 點號、空字串與全空白，保留有效名稱的前後空白、大小寫與 Unicode。

`project_device_coordinates(records, device_name=...)` 依 caller 指定的名稱讀取每列 `devices[device_name]` 的 `value` 與 `value_unit`。
它回傳分析用的 `dev_value/dev_unit` 欄位，保留其他內容、row index 與順序，不改來源表或 JSONL。
沒有 explicit finite flux 的列需要所選裝置的有限非 boolean 數值，缺失時錯誤列出 `devices[name].value` 與 row index。已有 explicit flux 時，缺裝置值以 NaN 表示，不阻擋解析。`value_unit` 是字串時原樣傳遞，否則投影為 None；只有 fallback frame 解析時才檢查單位是否匹配。
該列的 `devices` 不是 nested object 時，它讀取同一 record 攤平後的 `devices.<device_name>.value/value_unit` 欄位。它不猜第一台裝置，不從舊 flat 欄位 fallback，也不合成多裝置的 flux。
這是 opt-in 讀取工具，JSONL append 不呼叫它。Producer 的記錄格式見 [Record policy](../../../../zcu_lab/v2/measure-authoring.md#record-policy)。

## 作者 flux 換算

`flux_from_frame(table, fallback_frame=...)` 回傳同列序的 NumPy flux 陣列，不改表或記錄。
`table` 接受 nested records 與 dotted CSV 欄位。
每列依序使用 explicit flux、row frame、caller fallback。
有效 explicit flux 不檢查未使用的 frame 或 device。

Row frame 為 `flux_frame: {offset, period: {<device>: ...}}`。
它計算 `offset + Σ devices.<device>.value / flux_frame.period.<device>`，period 可帶負號。
使用 row frame 時，缺 offset 會報錯並指出列與 `flux_frame.offset`，不退回 fallback。
Row frame 不比對單位，因為值已使用該儀器校準的原生單位。

Nested 的 period keys 決定儀器集合，缺 device value key 時不貢獻。
Nested 明寫 NaN、inf 或 bool 都報錯。
Dotted 輸入以非 null 的 `flux_frame.period.<device>` 決定每列儀器集合。
Dotted 的 period 或 value cell 為 NaN 時，該儀器不貢獻；inf 與 bool 仍報錯。
攤平後無法分辨缺 key 與原始 NaN，只有 nested 輸入保證拒絕明寫 NaN。
需要檢查原始 NaN 的作者應使用 nested records。

Fallback 只收 `analysis.flux.FluxFrame`，單儀器與多儀器使用相同型別：

```python
from zcu_tools.analysis.flux import DevicePeriod, FluxFrame
from zcu_tools.resources.record_table import flux_from_frame

frame = FluxFrame(offset=-0.125, devices={
    "bias": DevicePeriod(period=0.8, unit="V"),
    "coil": DevicePeriod(period=-0.4, unit="A"),
})
flux = flux_from_frame(table, fallback_frame=frame)
```

Fallback 只在該列沒有 row frame 時使用。
每個有值儀器的 `value_unit` 必須與 `DevicePeriod.unit` 相同，否則報錯。
缺席儀器不需要單位。函式不換算單位、不讀參數檔，也不解讀舊 flat row frame。
Standalone v2 的 flat row 契約與單儀器 fallback 限制見下節。

## RecordTable v2 分析 frame（`schema.py`）

v2 定義 flat coordinate 契約。JSONL 儲存層不固定量測欄位 schema，也不自動轉換既有 CSV。呼叫端明確使用 `validate_record_table_v2` 或遷移函式時，才驗證完整 v2 座標。`resolve_record_flux` 只在使用各列的座標時檢查。Fluxdep utils 與 design search 使用此契約。Sample_merge 只攤平和串接，不套用 v2 座標驗證。座標欄位為 `flux`、`dev_value`、`dev_unit`、`flux_int`、`flux_period`（`RECORD_COORDINATE_COLUMNS`）。

**驗證（`validate_record_table_v2`）**：`dev_value`/`dev_unit` 必填，unit 為 A/V/native；coordinate 只接受 real numeric（拒絕 datetime/timedelta、boolean、complex），required value 必須 finite；`flux` 可空、只接受有限值或 null；`flux_int`/`flux_period` 成對出現，每 row 兩者同為有限或同為 null，period 必須 > 0；拒絕 legacy 別名欄位（`calibrated mA`、`calibrated A`、`Flux`、`flux_bias`）、重複欄位與孤立 frame 欄位。

**flux 解析（`resolve_record_flux`）**：closed per-row 來源順序 `explicit → row-frame → fallback-frame`，`sources` 為每 row 的權威來源、`explicit_mask` 只由 `sources` 推導；任一 row 無法解析即 fail-fast 列出 index。explicit flux 不要求裝置值或 frame。完整 row frame 只要求 finite `dev_value`、finite `flux_int` 與 positive finite `flux_period`，不限制未用的單位。沒有 explicit flux 時，只有一個非空 frame 欄位會報錯，訊息指出 row label 與缺少的欄名，不退回 fallback。兩個 frame 值都為空才可使用 fallback。Caller fallback 改收 `analysis.flux.FluxFrame`，必須恰有一台儀器，否則報錯並說明台數。這項檢查也適用於空表與 explicit-only 表。只有 fallback 的單位必須與該 row 的 `dev_unit` 一致，換算使用 offset 與帶號 period。未知與舊欄位不作 alias，也不觸發整表驗證。native 不隱含 A/V 換算；caller 的 frame 明示 native device value 到 flux 的映射。JSONL 寫入不計算 flux，notebook consumers 以 `JsonlRecordTable.read()` 取得最新 rows 後才解析座標。

**legacy 遷移（`migrate_record_table_v2`）**：caller 明確提供單一 A/mA/V/mV 來源單位，函式將 dev value 與宣告的 frame 縮放到 base A/V，回傳經驗證的 v2 DataFrame，不推論、不 alias，也不改寫輸入。離線的 [root migrations 工具](../../../../migrations/README.md) 提供公開轉換函式與 `python -m migrations` CLI。它明示改名有來源證據的舊量測欄位，重用這裡的座標轉換，並經 `ResultEntry.records(component)` 追加 JSONL。來源檔不變；無法確定的列不寫入，列入報告。工具不操作硬體，也不自行選擇真實資料。
