# 舊資料遷移 CLI 指南

**讀者：** operator、agent。使用離線 CLI 將一個舊 qubit 的 result／Database 遷移到新 entry，核對報告並處理重跑。

**Last updated:** 2026-10-27，component flux batch 轉換接線

本指南處理 `result/<chip>/<qubit>/` 與 `Database/<chip>/<qubit>/`。工具轉換支援的格式，將未支援的檔案與未知欄位原樣保留。報告以 `preserved` 表示原樣保全，以 pending 定位缺少必要轉換證據的支援項目。非法輸入與安全衝突會使操作失敗。

先依 [資料狀態對照表](../migrations/README.md) 選入口。Native 舊 component 不再走本工具，使用 [flux v1→v2→v3 升級](flux-migration.md)。Record 的舊值歸屬使用同一長期 CLI 的 [record 範圍](record-device-upgrades.md)。

Shipped mapping 共用 lab 的純 flux 轉換。每個完整 seed／value batch 先轉換，再以目前 `schema_version: 3` 驗證與發布，不先發布舊文件。現行校準為 `{offset, period}`，v1 自己的線保留 half 分支，coil 只保留 signed period，原 half／int 列報告。

每批與最終的 `flux_dev` 只保留明示來源，包含完整 overlay 中有效的 setup／mapping 繼承來源。無明示來源時留空，不從單位、live device 或佔位名稱猜測。缺名 local／coil 分別保留 `period.unnamed_device`／`period.unnamed_coil`，報告逐實際目的位置提醒改名，不因缺名列 pending。原資料明示的 `unnamed_*` 照明示處理。

Offset 每批按 v1 結構角色重算，自己的線優先，其次唯一 coil，否則 null。經 v2 中間形狀時也不靠佔位 `flux_source` 判斷角色，不新增 finalizer。Point 完整有效且明示來源的校準取代繼承 offset 與該儀器 period，其他 periods 保留。同批撞鍵、partial、非法名稱、數值與 spacing 仍拒絕。詳見 [flux 升級指南](flux-migration.md#storage-匯入的來源與覆寫)。

內部欄位對照與 publication 契約見 [storage migration 規格](../migrations/storage_migration/README.md)。獨立 CSV converter 的 API 與 CLI 見 [migrations README](../migrations/README.md#sample-csv--jsonl)。

## 目錄

- [確認路徑與授權](#確認路徑與授權)
- [先執行 dry-run](#先執行-dry-run)
- [正式執行與備份](#正式執行與備份)
- [核對結果與 pending](#核對結果與-pending)
- [中斷與完成後重跑](#中斷與完成後重跑)
- [差異與 exit status](#差異與-exit-status)
- [來源清理與後續使用](#來源清理與後續使用)

## 確認路徑與授權

停止會寫入來源與目的的 GUI、Notebook、agent 及其他 writer。工具不提供多程序 writer 協調或掉電交易保證。它只操作離線檔案，不操作硬體。

正式執行前，向使用者確認本次讀寫範圍，以及完成後刪除兩棵舊來源的授權。未取得刪除授權時，只執行 dry-run。逐項確認：

- `--repo-root` 是本次受保護的 checkout，所有路徑都使用已確認的絕對路徑。
- `--result-root` 與 `--database-root` 是舊來源根，`--entry-result-root` 是新 entry 根。新舊 result root 可以相同，但新 entry 與兩棵來源樹互不包含。
- `--source-chip`、`--source-qubit`、`--name` 各是一個合法路徑段。來源與目的路徑、`--qubit-kind` 都由操作者確認。
- `--backup-parent` 是 repo 外的備份根，與來源、目的互不重疊，也不包住來源。備份路徑及來源樹中的 symlink、非一般檔案會阻擋遷移。
- 新 entry 尚不存在，或由同一工具的初始化 ownership／manifest 擁有。工具拒絕認領任意既有 entry，即使內容看似相同。

應急工具或人工留下的半份 entry，先保存新舊兩邊的完整副本與 hash。將來源、目的、既有 ID 與差異交給維護者確認接手方式，保留已有有效資料，不刪 entry 強迫重跑。手動 `sha256sum` 清單與工具的 typed JSON 清單不是同一格式，不能當作 resume 依據。

## 先執行 dry-run

以下指令只規劃，不建立備份、entry、暫存檔或樣本表，不寫報告，也不刪來源。將範例路徑、kind、欄名與單位換成來源證據支持的值：

```bash
uv run --directory <checkout> --no-sync -- python -m migrations.migrate_storage \
  --repo-root <checkout> --backup-parent <external-backup-root> \
  --result-root <checkout>/result --database-root <checkout>/Database \
  --entry-result-root <checkout>/result \
  --source-chip <chip> --source-qubit <qubit> --name <new-entry> \
  --part all --qubit-kind qubit/fluxonium \
  --assume-flux-unit mA --assume-jpa-flux-unit V \
  --sample-value-column "calibrated mA" --sample-source-unit mA \
  --dry-run
```

`--part all` 同時處理 parameters 與 data，也可分開指定 `--part parameters`、`--part data`。Data-only 遇到尚未建立的 point 時，附屬檔保持 located pending；parameters 建 point 後再執行 data。

四個欄位／單位宣告各有用途：

| 選項 | 使用方式 |
| --- | --- |
| `--assume-flux-unit` | 只補 global 磁通組與 predictor 缺少的單位證據，報告標「假設單位」。不換算 component 數字。 |
| `--assume-jpa-flux-unit A\|V` | 獨立補 JPA 缺少的單位證據。缺此宣告就保持 pending，不借 global 假設。 |
| `--sample-value-column` | 指定 legacy CSV 的值欄。依來源填寫，不從欄名推論語意。 |
| `--sample-source-unit` | 獨立宣告 legacy CSV 的 A／mA／V／mV 座標單位，不借 global 假設。 |

V2 CSV 自帶 `dev_value/dev_unit`，不需 legacy CSV 宣告。假設單位只補缺證據，不能修補非法或互相衝突的明示單位。首次 dry-run 與正式執行使用相同宣告。

Native 轉換所需的歷史 evidence 用 `--run-evidence <file>` 提供。必要內容包括 experiment tag、cfg_type、完整歷史 cfg，以及 variable、axes、signal 的名稱、單位、dtype 與 shape。缺證據的 Labber 原檔仍保留並列 pending。開始時間與 run_id 可以未知；路徑日期、mtime、目前設定與裝置狀態都不能代填歷史事實。Evidence 欄位與驗證規則見[內部規格](../migrations/storage_migration/README.md#native-歷史證據)。

核對 stdout JSON report 中每個來源的去向、pending 原因、備份與目的路徑。確認後移除 `--dry-run`，其餘參數保持相同。

## 正式執行與備份

正式執行先完整備份兩棵來源樹，包含空目錄，再逐檔驗 SHA256，驗證通過才建立 entry 並發布資料。省略 `--backup-parent` 時使用 checkout 平行的 `<checkout-name>-backup/`。

```text
<external-backup-root>/
├── result/<chip>/<qubit>/
├── Database/<chip>/<qubit>/
└── migration-records/<chip>/<qubit>/
    ├── result.sha256
    ├── Database.sha256
    ├── migration-initialization.json
    └── migration-report-<UTC-date>.json
```

兩份 `.sha256` 是包含路徑、檔案 hash 與目錄的 typed JSON。多次遷移可共用備份根，工具只核對本次 chip/qubit 的鏡像。既有相同檔案略過，缺項補完；不同 bytes、額外檔案或型別衝突會阻擋寫入，既有內容不覆寫。

初始化只在 Result root 的已記錄暫存名稱建 entry，驗證後 rename 到正式名稱。Database 是後續輸出位置，首次 data publication 才建立所需目錄。工具不以空 Database 表示初始化完成。

## 核對結果與 pending

新 entry 位於 `result/<new-entry>/`，資料輸出位於 `Database/<new-entry>/`。逐項審閱報告與實際結果：

- setup、points、libraries 與分析 JSON 的值都有來源。磁通數字有明示單位或報告中的假設宣告。
- predictor、dispersive、T1 與 samples 已轉換，或逐項列 pending；未知欄位仍可在目的容器與報告指定的位置讀到。
- 每個 point 的圖位於對應 `figures/`。未支援的 result／Database 檔以原相對位置保全；報告的 `preserved` 不表示已轉 native 或可由分析工具解析。
- 每個 Labber 原檔都保存在 `Database/<new-entry>/Labber/<原相對路徑>`，bytes 不變。成功轉換者另有 `runs/<experiment>_<run_id>/data.h5`。
- 缺少開始時間時，native 保存 `started_at=None`、`completion=unknown`、`finished_at=None`。缺可信 run_id 時，報告標「遷移賦值」，分配時間不代表量測時間。
- Legacy CSV 有非法或缺證據的列時，合法列照常轉換，完整 CSV 另保存在新 entry 的 `samples.csv`。Pending 按來源列定位。
- Processed spectrum 的正式檔與 residual companion 分開列出，未轉 datasets／attrs 在 `spectrums.legacy.hdf5` 可見。

Pending 的重點是缺哪項證據，以及原文保在哪裡。未知但已完整保全的欄位不另列永久 pending。支援項目缺必要證據時仍列 pending，保全原文不能冒充轉換成功。

## 中斷與完成後重跑

中斷時保留 entry、工具暫存檔、manifest 與備份，直接重跑原指令。工具只補缺項，沿用 entry UUID、建立時間、run_id 與報告日期。`--resume` 仍可明填。兩部分分開執行時，第二次改 `--part` 即可續接同一 manifest。

重跑沿用首次的來源、entry 名稱、kind、四個欄位／單位宣告、mapping 版本與已固定的歷史 evidence。尚未固定 native run assignment 的 pending 項可補 evidence，固定後的內容則須一致。

完成後也可用原指令重驗。工具先驗完整備份，再比較 setup、points、libraries、分析 JSON、JSONL、processed 正式／companion、figures、raw、Labber 與 native。YAML／JSON 按 owner typed 語意比較，保留 unknown raw，排版差異不算內容變更。Binary／JSONL 比較已驗 SHA256。

相同內容不重寫、不追加、不重新分配 ID。報告的 `skipped` 逐項列出 `source`、`destination`、`comparison` 與 `summary`。這些重驗也可使用 `--dry-run`，維持零寫入。

## 差異與 exit status

| Exit | 含義與操作 |
| --- | --- |
| `0` | 執行成功，相同目的已略過。仍須逐項審閱 pending，exit 0 不表示全部檔案都已轉格式。 |
| `2` | 輸入或安全差異。Stderr 列來源、目的、比較種類與預期／實際摘要。保留兩邊內容，人工確認。 |
| `1` | 執行或 I/O 故障。保留現有狀態，排除故障後以原宣告重跑。 |

不同內容包含合法人工改值。差異診斷不覆寫目的、manifest 或原報告。需要保存 stdout／stderr 時，另存到獨立檔案，不指定受檢檔案作為輸出。

舊工具 manifest 缺文件 baseline 時會報「缺比較依據」並 exit 2。請維護者人工比對，或經使用者確認後在新備份上重新遷移。

## 來源清理與後續使用

Parameters 與 data 都完成、所有 publication 完成、沒有暫待 point files 或 failure 後，工具才重驗備份與完整來源清單並清理兩棵舊來源。永久 pending 可留下，報告位置改指備份。來源新增、變更或非工具搬移造成的缺檔都會阻擋清理。其他 qubit、上層 chip 及共用備份根保留。

自動報告保存在備份根的 `migration-records/<chip>/<qubit>/migration-report-<UTC-date>.json`。完成清理後，新 entry 的 `records/` 另存日期報告，同處的 `migration-state.json` 保存固定身分與比較依據。可用 `--report <external-file>` 選獨立外部報告位置，不能寫入來源樹或備份鏡像。

清理中斷後，原指令只收尾清理，允許已核實刪除的來源缺席。來源清理後重跑仍從已驗 backup／manifest 重驗目的。相同 baseline 的來源被放回時，工具驗全部目的後略過並保留放回的來源，不刪第二次；來源不同則報差異。

遷移未完成時保留固定 entry 名稱。完成後的 entry 改名只移動 Result，UUID 與 Database 舊輸出不變；以新名重新開啟後，新 Save 寫到新名稱的 Database。一般 entry 與改名契約見 [resources.entry](../lib/zcu_tools/resources/entry/README.md)。
