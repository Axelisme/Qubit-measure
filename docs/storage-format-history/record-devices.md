# Record device 值歸屬

**讀者：** operator、lab-author、framework-dev、agent。辨識無名舊值與具名 device 值，選擇明確升級與分析 fallback。

## 改前與改後

舊 JSONL 列用 flat pair，沒有 device 名稱：

```json
{"run_id":"a","dev_value":0.25,"dev_unit":"V","flux_int":0.1,"flux_period":0.8}
```

Operator 確認值來自 `bias`，用同一長期 `scripts/upgrade_parameters.py --scope records --device-name bias` 預覽，再加 `--apply`。結果為：

```json
{"run_id":"a","devices":{"bias":{"value":0.25,"value_unit":"V"}},"flux_int":0.1,"flux_period":0.8}
```

名稱是 operator 宣告，不是工具推測。名稱不得含 ASCII 點號、不得為空或全空白，其餘原樣保留。工具不換算單位，也不限制單位集合。

## 辨識與保留

JSONL 不加 header 或 row version。按列檢查舊 pair 與目標具名 pair，分類為 old、new、partial、corrupt。同檔可混合 old/new；任一 partial/corrupt 都拒絕整檔並報行號，其他檔案繼續。

目標只有 metadata 時可以合入 pair。舊 pair 與目標任一 value/value_unit 同時存在時拒絕，不覆寫。無 device 資料的列與其他 device entry 原樣保留，不用目前 schema 驗證未涉 payload。

Apply 先保存原 bytes 備份，再原子替換。列順序、重複 run_id、未知欄位與未涉資料不變。重跑跳過已升級資料。它不經 append，不補 `component` 或 `flux_frame`，也不改 component `schema_version`。

## Flux 與歷史名稱

現行 `flux_from_frame` 使用 `flux_frame: {offset, period: {<device>: ...}}`，不自動解讀歷史 flat `flux_int/flux_period` 或 v2 per-device frame。只有舊 flat frame 且沒有 explicit flux 的列，需要 caller 確認歷史校準與單位，再以新 `FluxFrame` 明示 fallback。已有 v2 per-device frame 須先在分析副本明確轉換。具名值的升級不合成歷史校準。

Device 改名或改接線不批次改寫歷史 records。歷史保留當時的名稱與 frame；目前 component 的處理與完整操作例子見 [Record 升級與改名改接線](../record-device-upgrades.md)。Component 版本鏈另見 [參數文件升級](../parameter-upgrades.md)。
