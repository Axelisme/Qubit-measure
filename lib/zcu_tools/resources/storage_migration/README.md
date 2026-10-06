# `zcu_tools.resources.storage_migration` 離線遷移

**Last updated:** 2026-10-06，具體 caller 與雙欄位 schema identity

本 package 組合 legacy result／Database 與新參數條目。正常 runtime loader 不使用它。公開入口是 `migrate_storage` 與 `load_run_evidence`，型別從 package root 匯入。

- 組合根提供 mapping、entry 的共用 kind 註冊及 experiment native validation callback。Lib 不 import `zcu_lab`、experiment 或舊資源容器。
- Native schema 與同步 validation 以 `(experiment tag, cfg_type)` 的明確歷史身分配對。缺 cfg identity 或無對應 schema 時只 pending，不回退 tag-only。
- Parameters 與 data 可分開執行。第二部分明確 resume 同一 manifest，不合併任意既有條目。完成的 setup／point 仍可編輯；resume 驗證當前文件，不恢復舊值。
- Manifest 保存固定 entry／run 身分、來源 baseline、完整 evidence raw 與逐檔 publication 狀態。Report 是累積快照。兩者保留同 major 的未知 JSON 欄位。
- Native publication 與驗證完成後才搬移對應 Labber。搬移先複製、驗證雜湊，再刪來源。缺歷史證據時保留來源並列 pending，不猜名稱或補量測時間。
- 原 result 樹不改動。未知樣品、圖片與 workflow 資產副本不是正常 runtime 資料。Module cfg 只列待處理，不用 generic YAML 冒充新 schema。

`state.py` 集中 report／manifest encoding 與 copy、native、move 共用的 publication recovery。它不提供跨檔交易、掉電保證、硬體操作或多程序 writer 協調。

`tools/migrate_storage.py` 是離線組合根。具體 mapping 與固定 native declarations 位於 `zcu_lab`，沒有動態 discovery 或新 runtime registry。Module cfg 的轉換仍歸後續 owner；本 package 只產生對照、待處理記錄與字串引用。操作授權、來源保全、pending／resume 與修改 kind 的步驟見 [Agent 離線遷移說明書](../../../../docs/storage-migration.md)。
