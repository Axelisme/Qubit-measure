# 離線遷移工具

本目錄提供使用者明確執行的舊資料轉換。正常量測 runtime 不 import migrations。工具不操作硬體，也不替使用者選擇來源檔或推論單位。

只為遷移存在的程式、對照表與宣告都放在本目錄，不放進 `zcu_lab` 或 `lib`。遷移完成後，刪除整個 `migrations/`（連同 `tests/migrations/` 與 `docs/storage-migration.md`）即可，不需要到其他目錄找殘留。使用者於 2026-10-14 表示：「我自己會預期，我遷移完後刪除migrations就可以了」。

## Legacy result／Database → native storage

`migrations.storage_migration` 擁有離線轉換、歷史 evidence 與可續跑 publication。Caller 明填 mapping 與 native 驗證，不接正常 runtime。`migrations.legacy_labber.load_legacy_labber_payload` 以明確 disk schema 讀取 single 或 marker-qualified grouped Labber。未標記多 log 以 caller 的 schema 變數順序對應 root、Log_2 等，各自嚴格驗證 label、單位與 shape，不猜 variable identity 或單位。

CLI 以 `python -m migrations.migrate_storage` 執行。操作授權、dry-run、pending 與 resume 步驟見 [Agent 離線遷移說明書](../docs/storage-migration.md)，框架邊界見 [storage migration README](storage_migration/README.md)。具體 mapping 在 `storage_mapping.py`，固定 native declarations 在 `migration_experiments.py`，兩者都只為遷移存在，跟本目錄一起保留或刪除。`lib` 與 `zcu_lab` 都不 import 本目錄。

`--result-root` 指定舊來源根目錄，`--entry-result-root` 指定新 entry 根目錄。兩者可同為 `<checkout>/result`，但新 entry 不得與本次來源樹互相包含。以下只做 dry-run，不建立備份或寫入目的地：

```bash
uv run --directory <checkout> --no-sync -- python -m migrations.migrate_storage \
  --repo-root <checkout> \
  --result-root <checkout>/result --database-root <checkout>/Database \
  --entry-result-root <checkout>/result \
  --source-chip <chip> --source-qubit <qubit> --name <new-entry-name> \
  --part all --qubit-kind qubit/fluxonium \
  --backup-parent <external-backup-parent> --dry-run
```

正式執行前，依遷移說明書核對報告與備份位置。預設備份累積在 checkout 平行的 `<checkout-name>-backup/`，保留 `result/<chip>/<qubit>/` 與 `Database/<chip>/<qubit>/`。共用根可以已存在，本次任一鏡像已存在就停止。逐檔 hash 清單與自動日期報告在 `migration-records/<chip>/<qubit>/`，不寫入來源鏡像。兩部分與安全清理完成後，entry 的 `records/` 另存日期報告。`--qubit-kind` 須符合來源元件，不由工具推論。

## Sample CSV → JSONL

`migrations.sample_csv.convert_sample_csv` 接受一或多份 CSV、已開啟的 `ResultEntry` 與元件名，追加到該條目的 `samples/<component>.jsonl`。來源位元組不變。公開函式回傳 `MigrationReport`，列出追加列數、待判斷列及原因、輸出路徑。

```python
from migrations.sample_csv import convert_sample_csv

report = convert_sample_csv(
    ["old.csv"],
    entry=entry,
    component="Q1",
    dev_value_column="calibrated mA",
    source_unit="mA",
)
```

v2 CSV 已有 `dev_value` 和 `dev_unit`，不需提供 legacy 參數。舊格式必須明填來源欄位與 A／mA／V／mV；標題的單位文字不作推論。座標驗證與縮放重用 sample table owner。已知 measurement 欄位改名；未知欄位、非法值或缺少宣告使整列待判斷，不寫入。

合成 key 為 `legacy-<Path.stem>-<資料列號>`。列號從 1 起算，不計 header。來源為 manual，point 預設為 null，可明填。同次輸入 stem 重複時，寫入任何列前拒絕。再次執行會追加相同 key 的歷史，讀取時取最後一列，不提供批次 rollback 或並行 writer 保證。

CLI 使用既有 Result／Database 條目：

```bash
uv run --directory <checkout> --no-sync -- python -m migrations old.csv \
  --result-root <result-root> --database-root <database-root> \
  --entry <entry-name> --component Q1 \
  --dev-value-column "calibrated mA" --source-unit mA
```

CLI 在 stdout 輸出 JSON report。有待判斷列時 exit 1，全部轉換成功時 exit 0。I/O 與設定錯誤直接回報。使用者應先核對合成資料試轉結果，再自行安排真實資料遷移。
