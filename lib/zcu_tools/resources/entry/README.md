# `zcu_tools.resources.entry` — result entry composition

**Last updated:** 2026-10-04 — create/open 與 setup metadata 切片

`ResultEntry` 組合兩個明確傳入的根目錄。名稱是安全的單一路徑段，不代表物理量或身分。`setup.yaml` 的 UUID `entry_id` 是身分，建立後不可變。載入驗證 UUID 與 UTC 建立時間；既有 handle 不接受 refresh 帶入另一個身分。

`ResultEntry.create` 建立新格式 setup、points 與 records 目錄，以及同名 Database 目錄。任一目的地已存在就拒絕。建立失敗只清本次建立的條目目錄，不刪 caller 的根目錄或既有資料。`open` 要求完整的新格式條目，不猜測 legacy 格式。

`SetupView` 讀取 DocumentStore 的記憶體快照。單行 description 寫入與 `EditView` 都使用同一個型別化交易。DocumentStore 擁有單檔衝突檢查、版本、SI／工作單位邊界與 `.entry.lock`；ResultEntry 擁有身分的提交前驗證。

`rename_entry` 移動兩個目錄，不改檔案內容。第二次移動失敗時復原第一次；復原也失敗則以 `PartialCommitError` 回報已完成、待完成與復原失敗的路徑，並保留兩個原因。這不是跨檔掉電保證。

本模組尚未接線到 ContextService、notebook caller、GUI 或 MCP。元件 kind/schema、完整屬性視圖與後續的工作點、來源和角色解析由同一組新 entry 模組逐步提供，不改現行 context 的責任。
