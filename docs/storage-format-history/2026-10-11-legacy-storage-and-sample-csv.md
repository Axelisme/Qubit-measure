**讀者：** framework-dev、agent。辨識舊 result／Database 與 sample CSV，查找 native entry、record JSONL 的目的形狀與離線遷移工具。

# 舊 result／Database 與 sample CSV 遷入 native entry

日期為 2026-10-11，作為這組離線工具的歷程摘要。Git 的 `--follow` 歷史顯示，storage CLI 在 2026-10-06 的 `56cd6c22f` 落地，於 2026-10-11 的 `b41f0153f` 移入根 `migrations/`。Sample CSV converter 在同日的 `06465058f` 落地。本篇整理目前工具的來源與目的形狀，不表示後續的 Container 與 record 命名已在 10-11 全部存在，也不追溯更早歷史。

範圍是舊 `result/<chip>/<qubit>/`、`Database/<chip>/<qubit>/` 到具名 entry，以及 sample CSV 到 record JSONL 的 v2 分析欄位。不涵蓋 flux 格式與後續 record 的 flux_frame／component 變更。

## 改前形狀

舊 result 以 chip／qubit 分層，參數分散在根 `meta_info.json`、`params.json`、`module_cfg.yaml` 與工作點子目錄。工作點可位於 `exps/<sub>/` 或舊直屬 child。樣本放在根 `samples.csv`。Database 在同一組 chip／qubit 下保存 Labber HDF5，而不是 native run 容器。

以下檔案形狀取自 [mapping 表](../../migrations/storage_migration/README.md#來源檔案與目的)，尖括號為該文件的路徑參數：

```text
result/<chip>/<qubit>/meta_info.json
result/<chip>/<qubit>/params.json
result/<chip>/<qubit>/module_cfg.yaml
result/<chip>/<qubit>/exps/<sub>/meta_info.json
result/<chip>/<qubit>/samples.csv
Database/<chip>/<qubit>/<來源相對路徑>
```

具體的 Labber 例子來自 [migration fixture](../../tests/migrations/storage_migration/fakes.py) 的 `write_run`：

```text
Database/chip/qubit/2025/10/scan.hdf5
```

它含 signal `[1+2j, 3+4j]`，單位 V，x 軸 `[0.0, 1.0]`，單位 s。Comment 的 cfg 為 `{"frequency": 12.5}`，tag 為 synthetic_scan。這是測試中的合成資料，不是真實量測。

舊 CSV 不一定明示座標單位。例如 [sample publication 測試](../../tests/migrations/storage_migration/test_params_samples.py) 的來源原文為下列內容，檔案使用 CRLF：

```csv
calibrated mA,T1 (us)
2,50
bad,60
```

V2 CSV 已改為 `dev_value`／`dev_unit`，但仍是交換檔，不是 JSONL 歷史。[CSV converter 測試](../../tests/migrations/test_sample_csv.py) 中也有這份來源：

```csv
dev_value,dev_unit,T1 (us),comment
0.1,V,50,raw
```

## 改後形狀

新 entry 在 `result/<entry>/` 保存 setup、points、libraries、分析文件與 records。Database 的 `<entry>/` 是輸出位置，不是 entry 身分的唯一依據。主要對照如下，沿用上述 mapping 表的範本：

| 舊內容 | 新 entry／輸出位置 |
| --- | --- |
| 根 meta_info | `result/<entry>/setup.yaml` |
| 工作點 meta／library | `result/<entry>/points/<sub>/point.yaml`、`module_cfg.yaml` |
| 根 module_cfg | `result/<entry>/module_cfg.yaml` |
| params 的已知 fit | `result/<entry>/predictors/<Q>.json`、`dispersive/<Q>.json`、`t1_curve/<Q>.json` |
| 樣本 | `result/<entry>/records/components/<Q>.jsonl` |
| Labber 原檔 | `Database/<entry>/Labber/<來源相對路徑>` |
| 可轉換的量測 | `Database/<entry>/runs/<experiment>_<run_id>/data.h5` |

[歷史 identity 測試](../../tests/migrations/storage_migration/test_run_history.py) 為同一個 scan 明示可信的 run_id `20251001T000000Z-abc123`，目的 entry 名為 destination。轉換後同時保留原 Labber 與新增 native：

```text
Database/destination/Labber/2025/10/scan.hdf5
Database/destination/runs/synthetic_scan_20251001T000000Z-abc123/data.h5
```

Native 目的目前使用 zcu.container major 3。陣列與具名軸在 `vars`，實驗 cfg 與 run metadata 在 `docs/cfg`、`docs/metadata`，保留的 producer 子樹在 `groups`。Cfg 另帶 cfg_type 與 schema_version。這些是現行工具的輸出，不應將早期 native 2.x 當作同版 Container 讀取。

上述 `calibrated mA` CSV 在明示欄位與 mA 單位後，合法首列轉成 `records/components/Qlegacy.jsonl`。以下為同一 publication 測試結果的 JSONL 欄位節錄，省略由儲存層產生的 UTC `at`：

```json
{"run_id":"legacy-samples-1","point":null,"source":"manual","dev_value":0.002,"dev_unit":"A","t1_us":50}
```

第 2 列的 bad 不寫入樣本表，報告列 pending，完整來源另保存在新 entry 的 `samples.csv`。JSONL 是只追加歷史，讀取取每個 run_id 的最後一整行，不合併較早欄位。它與 `records/ledger.jsonl` 的事件行不是同一格式。

V2 分析 frame 使用明示 `dev_value`／`dev_unit`，單位為 A、V 或 native。Legacy converter 將宣告的 mA／mV 座標縮放成 A／V。已知 measurement 欄位改為工作單位名稱，例如 `Freq (MHz)` → `freq_mhz`、`T1 (us)` → `t1_us`、`T1err (us)` → `t1_err_us`，不將數值換成 Hz 或 s。JSONL 儲存層本身不強制 v2 schema，v2 是呼叫端選用的分析契約。

## 為什麼改

舊目錄、Labber comment 與 CSV 欄名不能各自提供完整的 typed cfg、歷史身分、單位及完成狀態。遷移需要把已知資料放進各 owner 的文件，並將缺少的歷史證據列出，不能把目前設定或路徑日期當成當年的事實。

Native `data.h5` 保存實驗的全部 Data Variables 與 cfg／metadata，Labber 留作衍生輸出與原檔保全。Entry 將 setup 範本、獨立 point 與記錄分開。JSONL 保存追加歷史，CSV 留作匯入／匯出；明示的 v2 座標讓分析端不必從 `calibrated mA` 這類欄名猜單位。這些目的不同，因此沒有讓正常 runtime 自動猜格式、雙讀 CSV 或執行遷移。

## 如何辨識

- 舊 result 通常有 chip／qubit 階層、meta_info.json、params.json 或 samples.csv。Native entry 則有 setup.yaml 與 points，record table 位於 `records/components/`。混合目錄可能是部分遷移，不能只憑新目錄存在就判完成。
- 檢查 HDF5 根屬性與結構，不靠副檔名或日期。現行 native 為 `format="zcu.container"`、`format_version` major 3，並使用 vars／docs／groups。Labber 依 log、channel、comment 等 schema 讀取，不因移入 Labber 目錄就變成 native。
- CSV 的 `dev_value,dev_unit` 是 v2 來源的線索。只有 `calibrated mA` 等舊欄位時，必須另給欄位與單位宣告。JSONL 則是每行一個 JSON object，含 run_id、point、at、source；不能把 CSV 改副檔名當轉換。
- Pending、preserved 與 converted 分開判讀。原檔已保全不表示已成功轉成 native，也不表示可供分析工具解析。以 migration report、manifest、備份與目的內容交叉核對完成程度。

## 遷移

整棵來源使用 `python -m migrations.migrate_storage`。依 [操作指南](../storage-migration.md) 指定 repo、舊 result／Database、新 entry result root、chip／qubit、entry 名稱、component kind 與 repo 外備份根，先加 `--dry-run`。Legacy CSV 另給 `--sample-value-column` 與 `--sample-source-unit`。Native 轉換需要的歷史 cfg、tag、variable／axis 名稱、單位、dtype 與 shape 由 `--run-evidence` 提供，不從檔名或目前裝置補造。

只有 CSV 時，使用 [migrations.sample_csv](../../migrations/README.md#sample-csv--jsonl) 的 `convert_sample_csv`，或 `python -m migrations`，明填 entry 與 component。Legacy CSV 必須給 `dev_value_column`／`source_unit`，v2 CSV 不需這兩個 legacy 宣告。Converter 不改來源 bytes。同 synthetic run_id 且內容相同的列重跑時略過，不同則拒絕，不重複追加。

未知 CSV 欄位保留原名原字串。與 producer metadata 或 canonical 名稱衝突時存入 `legacy_fields`，報告記錄 literal destination keys。缺證據或非法列保持 pending，不用預設值湊成成功。缺歷史時間時 native 保留未知時間與 completion=unknown；遷移分配的 run_id 不代表量測時間。

正式 storage migration 會建立並驗證外部備份，以 manifest 追蹤初始化與發布。停止所有 writer，取得讀寫及來源刪除授權後才移除 dry-run。來源清理與中斷續跑條件以操作指南為準，不能刪除半成品 entry 強迫重跑。正常 runtime 不提供舊來源的 fallback loader。

## 相關 ADR

- [ADR-0063](../adr/0063-persistence-ownership.md) 區分 entry、實驗資料、record history 與 ledger 的保存權威，以及離線遷移邊界。
- [ADR-0080](../adr/0080-datafile-generic-container.md) 定義現行 native Container 3 的表示法與 major 版本辨識。
- [ADR-0072](../adr/0072-module-boundaries.md) 將離線遷移程式隔離於根 migrations，不接入 runtime。
