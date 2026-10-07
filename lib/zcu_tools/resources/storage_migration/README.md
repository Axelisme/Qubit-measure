# `zcu_tools.resources.storage_migration` 離線遷移

**Last updated:** 2026-10-07，point-local typed library 轉換

本 package 組合 legacy result／Database 與新參數條目。正常 runtime loader 不使用它。公開入口是 `migrate_storage` 與 `load_run_evidence`，型別從 package root 匯入。

- 組合根提供 mapping、entry 的共用 kind 註冊、planning 前的 concrete cfg validation 與發布後的 native validation callback。Lib 不 import `zcu_lab`、experiment 或舊資源容器。
- Native schema 與兩個 callbacks 以 `(source tag, cfg_type)` 的明確歷史身分配對。缺 cfg identity、無對應 schema 或 typed-invalid cfg 時只 pending，不回退 tag-only。`native_tags` 只轉換新 RunMetadata 的 tag，不改歷史 raw 或正常 loader。
- Parameters 與 data 可分開執行。第二部分明確 resume 同一 manifest，不合併任意既有條目。完成的 setup／point 仍可編輯；resume 驗證當前文件，不恢復舊值。
- Manifest 保存固定 entry／run 身分、來源 baseline、完整 evidence raw 與逐檔 publication 狀態。Report 是累積快照。兩者保留同 major 的未知 JSON 欄位。
- Native publication 與驗證完成後才搬移對應 Labber。搬移先複製、驗證雜湊，再刪來源。Source_removed 只驗證目的 hash，舊位置重用不改既有 run，也不刪新檔。缺歷史證據時保留來源並列 pending，不猜名稱或補量測時間。
- 整個舊 result_root 不改動；首次與 resume 都拒絕會寫入或刪除該樹的配置。未知樣品、圖片與 workflow 資產副本不是正常 runtime 資料。
- Parameters 將 child point 的 module／waveform 轉成 `cfg_model` 的正式 Spec，透過 `ResultEntry.module_library` 提交 point-local library。Mapping 明示來源 family 與 canonical key，typed reference remap 由 `cfg_model` 擁有。Root library 沒有 point 歸屬，保留 located pending。
- 未支援 expression、Spec、缺 mapping 或 child reference 不能轉換時保留來源與定位，不 evaluate。Arb asset key 保持原值，不搬 asset，也不保證全部模組已可 materialize。
- Library dry-run 驗證隔離候選，不取得目的 handle。其 report rows 保留 canonical `new_path`，`new_file=None` 本身不表示 pending。正常提交後 manifest 保存來源與目的 hash，resume 核對已記錄完成的 library 而不重做。

`state.py` 集中 report／manifest encoding 與 copy、native、move 共用的 publication recovery。ModuleLibrary 的單檔提交與 manifest 不是跨檔原子。Library 已寫入但未記錄完成的窗口沒有專門 recovery，需取得清理授權後刪目的地重跑。本 package 不提供掉電保證、硬體操作或多程序 writer 協調。

`tools/migrate_storage.py` 是離線組合根。具體 mapping 與固定 native declarations 位於 `zcu_lab`，沒有動態 discovery 或新 runtime registry。Parameters 模式不執行 data 搬移或來源刪除。操作授權、來源保全、pending／resume 與修改 kind 的步驟見 [Agent 離線遷移說明書](../../../../docs/storage-migration.md)。
