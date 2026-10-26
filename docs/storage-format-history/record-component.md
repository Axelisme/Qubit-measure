# Record component 信封

**讀者：** operator、lab-author、framework-dev、agent。辨識 record 列的 component 欄位與舊資料的讀取規則。

## 改前與改後

元件歷史仍放在 `records/components/<component>.jsonl`。例如 `Q1.jsonl` 的舊列只有檔名標示元件：

```json
{"run_id":"run-a","point":"bias","at":"2026-10-25T00:00:00Z","source":"manual","t1_us":20.0}
```

新 append 在同樣的列中加入 `component`：

```json
{"run_id":"run-a","point":"bias","at":"2026-10-25T00:00:00Z","source":"manual","t1_us":20.0,"component":"Q1"}
```

## 原因與寫入契約

複製或匯整單列後，原檔名不一定還在。列內的 `component` 保留寫入時的元件名稱。

`ResultEntry.records(component)` 把元件名綁定給 `JsonlRecordTable`。所有 append 都以 binding 的名稱填信封，producer 同名欄位不能覆寫它。直接建立 table 的 writer 必須傳 `JsonlRecordTable(path, component="Q1")`，temporary path 也不例外。未傳 component 的 table 只能讀取與匯出，append 在寫檔前拒絕缺名或空名。

## 辨識與舊資料

JSONL 沒有獨立版本標頭。新 writer 每列都帶非空字串 `component`。缺此欄的舊列仍可用 `read()` 與 `latest()` 讀取，reader 不從 binding 或檔名補值。歷史檔可以同時包含新舊列；判讀以列為單位，不能以檔名或某一列推斷整檔格式。

不提供舊 records 的補寫或遷移工具，也不重寫既有歷史。這項變更不新增 flux frame 或參數 schema 規則。既有 CSV 遷移工具寫新列時也帶 binding 的 component；來源同名欄位保存在 `legacy_fields.component`，不當作元件身分。

## 相關契約

- [Record table owner](../../lib/zcu_tools/resources/record_table/README.md) 定義 append、每個 run 的最新整行讀取與 CSV 匯出。
- [ADR-0063](../adr/0063-persistence-ownership.md#保存目的與-owner) 定義 record 與事件 ledger 的保存權責。
- [離線 CSV 遷移](../../migrations/README.md#sample-csv--jsonl) 定義來源欄位保留與重跑比對。
