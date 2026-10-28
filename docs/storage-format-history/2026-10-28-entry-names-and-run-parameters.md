# Entry 名稱與量測參數快照

**讀者：** operator、lab-author、framework-dev。辨識 entry UUID 移除、Run 參數快照及重存目的地的變更。

## Entry 歸屬

同一 Result root 下，entry 以資料夾名稱歸屬。新 setup、point、ledger 事件與 Run metadata 不再產生 entry UUID。Run ID 與 event ID 保持原契約。

Setup／point 的 general 保留 created_at、description 與 ext。EntryIdentity 可固定 UTC 建立時間，不含 UUID。舊 entry_id 缺少、值不合法或形狀不同都不影響讀取，read／refresh 不改原檔 bytes。明確修改時移除舊欄位，其他參數及容器擴充欄位沿 owner 的保存規則保留。

開啟中的 setup 同名換檔後，公開 refresh、new_point 或重新 open 使用新內容，不以 UUID 拒絕。這不新增 setup 自動監看。Point source、owning roots、安全路徑段與 symlink 邊界仍要核對。

Ledger 的歷史 entry_id 不參與投影或 append。歷史 ImportPayload 不再解讀 source_entry_id，source_event_id、source_event_json 及原始 JSONL 行保持。Migration 的 cloned_from evidence 只解讀 point，原始 identity 欄位留在 raw evidence，不改名為操作 token。

## Native Run

Container major 仍為 3。新 metadata.snapshot 保存 entry_name、point 與 roles。舊 metadata 沒有 entry_name 時不從 UUID、檔名或所在目錄猜名稱。

新 parameter_snapshot 保存當次 capture 的 point 參數、library Specs、authored aliases 與固定名稱。這份環境不包含持久 target 或 storage。GUI Run 在 admission 時固定內容，Save 不重新捕捉目前 point。

Load 接受任意位置的 data.h5，只從檔內還原參數環境。外部 point 改動、刪除或換 entry 都不影響 Analyze／Post。缺快照的舊 Run 使用空環境，不製造 general、created_at 或 SavedPoint。分析可以使用自己的預設值。

未使用的欄位與區塊延後驗證。使用壞參數、Spec、alias 或 metadata 時，owner 指出來源與欄位，不把壞快照降成空內容。Save、完整投影與 edit 仍執行完整驗證。Header／major、component schema_version、JSON／YAML 表示與實際使用的 path 安全規則不變。

## Save 與寫回

GUI Save 使用目前 result／Database roots 加上 metadata 記錄的 entry、point 及 caller stem，不開 ResultEntry、不查 entry／point 存在性，也不沿原檔路徑或 live selection 猜目的地。

完整 data.h5、analysis JSON 與 Run 圖片寫到 Database/<entry>/runs/<stem>/。Result 圖副本寫到 result/<entry>/points/<point>/figures/。缺 entry 名稱使用 unnamed_entry，缺 point 使用 unnamed_point。Writer 建立所需目錄，同名 native、JSON 與 PNG 覆寫；Labber 仍沿既有 numeric suffix 規則。

無 storage 的 Run 保留 artifact/member declarations，首次 Save 前 outputs 可以為空，但不表示已 Save。Save report 發布實際 paths 與每項成功或失敗。Bound Run 保留既有預覽；同 session 改名或改 roots 時，預覽可能不同於實際 Save 目的地。這個限制不影響實際輸出 binding。

Writeback 的歷史內容供提案求值及 alias／slot 路徑使用。Preview／Apply 寫到命令當下 GUI 選定的 point，不寫回歷史來源。未選 point 時拒絕，空歷史環境不是拒絕理由。

## 離線 migration

MigrationIdentity 與 MigrationReport 使用 migration_id 保護一次遷移操作。初始化暫存名稱使用 .migration-operation-<migration_id.hex>。這個 token 不用於 entry 歸屬，不寫進 setup、point、Run 或 ledger。

新版不讀舊 ownership 欄位 entry_id，也不提供 fallback 或自動轉換。需要重跑或 cleanup 的舊 manifest、report 或 initialization 含此欄位時，操作拒絕並指名檔案。請用舊版完成，或確認狀態後自行處理。工具不自動刪改來源、備份或目的資料。

備份核對、錯誤 owner、越界路徑及 cleanup 保護維持原契約。其餘 manifest/report schema 不重設。

責任與取捨見 [ADR-0063](../adr/0063-persistence-ownership.md)。操作程序見 [migration 指南](../storage-migration.md)。
