# `migrations.storage_migration` 離線遷移

**Last updated:** 2026-10-20，exps points、entry predictors 與來源清理

本 package 組合 legacy result／Database 與新參數條目。正常 runtime loader 不使用它。公開入口是 `migrate_storage` 與 `load_run_evidence`，型別從 package root 匯入。

- 組合根提供 mapping、entry 的共用 kind 註冊、planning 前的 concrete cfg validation 與發布後的 native validation callback。本 package 不 import `zcu_lab`、experiment 或舊資源容器。`lib` 不 import `migrations`。
- Native schema 與兩個 callbacks 以 `(source tag, cfg_type)` 的明確歷史身分配對。缺 cfg identity、無對應 schema 或 typed-invalid cfg 時只 pending，不回退 tag-only。`native_tags` 只轉換新 RunMetadata 的 tag，不改歷史 raw 或正常 loader。
- `exps/<sub>/` 與舊直屬 child 都轉成同名 point，沿用 parameters mapping 與 pending 規則。兩種來源同名就拒絕，報告列出兩個來源，不合併或覆寫。
- `exps/<sub>/image/*.png` 經 copy、hash、manifest 與 resume 發布到 `result/<entry>/points/<sub>/figures/<原檔名>`。Data-only 不建立缺少的 point，而將圖片列為 located pending；parameters 建 point 後可 resume data 完成發布。`exps` 外的 image、samples 與 autofluxdep 資產維持 preserved 行為。
- `params.json` 以本次建立的唯一 qubit component 為對象，找不到唯一對應就整份 pending。Converter 先將原文區塊與 metadata 存入 setup 的 `component.ext.legacy_params`，再建 points，讓 points 繼承完整 YAML。根層與各 point 的 meta_info 值優先，params 只補缺；衝突保留 meta_info 值，並報告兩來源與 pending。
- `fluxdep_fit` 寫入 entry 的 `predictors/<component名>.json`，不寫 component fit 欄位，也不複製到 points。Predictor 格式 owner 是 `resources.predictor_file`。此 converter 在該 owner API 可用前直接寫契約規定的 JSON。檔案使用 `format=zcu.predictor`、整數 `version=1`，params 的欄名與有限數值遵循 simulate Params.from_mapping，完整性與物理限制交給 build_predictor。
- 舊 EJ/EC/EL fit 寫 `model=fluxonium`、`flux_bias=0.0`，`flux_int` 只作 alignment，不寫 predictor。舊 owner 未定義的 transmon 或明示 model fit 保持 pending。單位只讀根層或 fit 的明示 `flux_unit`；缺少或衝突就寫 null 並列 pending，不猜測或換算。`source.run_id` 固定 null；`source.created_at` 只接受可證實且含時區的 ISO 8601，否則 null 並列 pending。這三個 nullable 欄位仍必須存在；報告保留備份內原文定位。
- Parameters 與 data 可分開執行。第二部分明確 resume 同一 manifest，不合併任意既有條目。完成的 setup／point 仍可編輯；resume 驗證當前文件，不恢復舊值。
- Legacy evidence 1.x 的 params 與來源稽核使用本 package 的 `LegacyParameterSnapshot`、
  `LegacyParameterSource`、`LegacyCloneOrigin`。Wire shape 不變，完整 raw evidence 留在 manifest。
  Converter 產生 native 2.0 簡單 context，不把歷史 params 投影到 native。
- Manifest 保存固定 entry／run 身分、來源 baseline、完整 evidence raw 與逐檔 publication 狀態。Report 是累積快照。兩者保留同 major 的未知 JSON 欄位。
- Native publication 與驗證完成後才搬移對應 Labber。搬移先複製、驗證雜湊，再刪來源。Source_removed 只驗證目的 hash，舊位置重用不改既有 run，也不刪新檔。缺歷史證據時不產生 native 或逐檔搬移，只列 pending，不猜名稱或補量測時間。兩部分都完成後，這些舊檔隨來源清理，只保留在外部備份。
- 新 entry 可與舊資料共用 result／Database root，只要目的與本次兩個舊來源樹互不包含。目的已存在就拒絕，不覆寫其他 entry；report 也不得寫入本次來源樹。
- 正式執行在建立 entry 前完整複製本次舊 result 與 Database 到 repo 外的 `<name>-backup`。`backup_parent`／CLI `--backup-parent` 指定外部上層，預設為工具 checkout 的上層。`repo_root`／`--repo-root` 可明示被保護的 checkout。備份內 `result/` 與 `Database/` 保留原相對路徑與空目錄，`sha256.json` 列出每檔 SHA256。既有備份目錄、repo 內或與來源／目的重疊的路徑一律拒絕。Symlink 與非一般檔案不支援，拒絕後不進入遷移。
- `backup.py` 擁有完整複製、清單驗證與清理。Manifest 綁定清單 SHA256 及來源完整 hashes；每次 resume 驗證既有備份，不重做。已搬移或已清理的來源不必存在。未完成的備份不授權任何遷移；首次備份之後、entry 建立之前若中斷，須人工核對並移走該備份才可重新開始，工具不認領既有目錄。
- Parameters 與 data 都完成、無未完成 publication、無 `point figures` 暫待項且無 failure 後才清理。本次舊 `result/<chip>/<qubit>/` 與 `Database/<chip>/<qubit>/` 會刪除。永久 pending 可留下，但報告位置改指向備份。未知樣品、圖片與 workflow 資產副本不是正常 runtime 資料。
- 清理前重驗備份與完整來源清單。來源新增、變更或非工具搬移造成的缺檔會拒絕清理。先持久化 `cleaning`，再逐檔核對 hash、刪檔並移除空目錄。中斷後 resume 只收尾清理，允許已刪的備份檔缺席，不重跑轉換。不承諾並行 writer 或掉電交易；來源有其他 writer 時不得執行。Dry-run 只規劃，連備份目錄與清單都不建立。
- Parameters 將 child point 的 module／waveform 轉成 `cfg_model` 的正式 Spec，透過 `ResultEntry.module_library` 提交 point-local library。Mapping 明示來源 family 與 canonical key，typed reference remap 由 `cfg_model` 擁有。Root library 沒有 point 歸屬，保留 located pending。
- 未支援 expression、Spec、缺 mapping 或 child reference 不能轉換時保留來源與定位，不 evaluate。Arb asset key 保持原值，不搬 asset，也不保證全部模組已可 materialize。
- Library dry-run 驗證隔離候選，不取得目的 handle。其 report rows 保留 canonical `new_path`，`new_file=None` 本身不表示 pending。正常提交後 manifest 保存來源與目的 hash，resume 核對已記錄完成的 library 而不重做。

`state.py` 集中 report／manifest encoding 與 copy、native、move 共用的 publication recovery。ModuleLibrary 的單檔提交與 manifest 不是跨檔原子。Library 已寫入但未記錄完成的窗口沒有專門 recovery，需取得清理授權後刪目的地重跑。本 package 不提供掉電保證、硬體操作或多程序 writer 協調。

`migrations/migrate_storage.py` 是離線組合根。具體 mapping 與固定 native declarations 位於同目錄上層的 `migrations/storage_mapping.py` 與 `migrations/migration_experiments.py`，沒有動態 discovery 或新 runtime registry。單獨 Parameters 模式不執行 data 搬移；只有同一 manifest 的兩部分都完成才可清理，包含第二部分以 parameters resume 的情況。操作授權、來源保全、pending／resume 與修改 kind 的步驟見 [Agent 離線遷移說明書](../../docs/storage-migration.md)。
