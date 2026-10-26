# Component 參數文件升級

**讀者：** operator、lab-author、agent。辨識 component 文件版本，預覽與套用升級，並從備份還原。

## 版本與目前限制

`setup.yaml`、每個 `points/<label>/point.yaml` 與 entry template 都有整份文件的 `schema_version: N`。它描述 component schema。`format` 與 `format_version` 仍描述 YAML 容器，兩種版本不互相代替。

GUI／MCP 只讀 lab 宣告的目前版本。缺版或不同版本會在讀取處報錯，列出來源路徑、文件版本、目前版本與 upgrade 命令。讀取不改檔。不要為了消除錯誤手改版本標頭。

目前 lab runtime 宣告版本 1，離線升級鏈已提供 flux v1→v2，目標為 2。這個過渡階段只在副本驗收，runtime 1 不能讀 v2 產物。Runtime 移除舊欄位並升為 2 後，才對正式資料安排升級。已是 v2 的文件先完整驗證，再顯示 `skipped`。未標版文件只有通過完整舊格式分類後才視為 v1，不需要 operator 補標頭。四類範例、缺來源的手動處理與參考點假設見 [flux 升級指南](flux-migration.md)。

## 預覽與套用

從 checkout root 執行，使用該 worktree 的 interpreter。給一個或多個明確 YAML 路徑，shell glob 可選多個 point。

```bash
uv run --directory /absolute/checkout --no-sync -- python scripts/upgrade_parameters.py \
  /data/result/device/setup.yaml /data/result/device/points/*/point.yaml
```

預設是 `dry-run`。輸出按完整檔案路徑列出結果，point 路徑含 label。每筆欄位差異顯示 literal key tuple、原值與新值，`<missing>` 與 explicit null 不同。Dry-run 不建檔、不建鎖、不備份。

先關閉會寫這些文件的工作階段，審閱差異後才加 `--apply`。

```bash
uv run --directory /absolute/checkout --no-sync -- python scripts/upgrade_parameters.py \
  /data/result/device/setup.yaml /data/result/device/points/*/point.yaml --apply
```

每份合法候選先取得同名 `.lock`，鎖最多等待 5 秒。Runner 在鎖內重讀來源並驗證整條鏈，再用獨立檔名保存完整原 bytes。備份路徑是原檔名加唯一識別碼與 `.bak`。備份成功才以同目錄 temporary 檔原子替換來源。輸出列出備份路徑與 `applied`，替換失敗也保留並列出已完成的備份。

`rejected` 會列出未通過的檢查。該文件不替換，其餘文件照常處理。有任何拒絕，命令回傳 1；全部成功或跳過，回傳 0。已是 current 的合法文件不重寫、不再備份，重跑結果是 `skipped`。這不是多檔交易，也不保證掉電後整批一致。

若檢查指出混合版本或不完整欄位，先保留原檔與診斷，依 lab 的對應格式說明修復，再重跑 dry-run。只改版本數字不會補齊欄位，也不能證明文件可讀。

## 備份還原

關閉使用這份文件的工作階段，保留失敗後的文件作比對。將輸出所列 `.bak` 複製回原 YAML 路徑，再重新開啟或明確 reload。備份保留原版，因此目前 runtime 可能仍要求 upgrade。不要把還原當成已完成升級。

## 同一 CLI 的 record 範圍

`--scope records --device-name <name>` 逐列處理 JSONL 歷史，不使用 component 版本鏈。名稱由 operator 宣告，不從 setup 或歷史內容推測。工具將完整的舊 `dev_value/dev_unit` 移入具名 `devices`，保留所有歷史行與未涉 payload，不補 frame/component。單檔有部分或腐敗列就整檔拒絕，其他檔案繼續。

四類形狀、dry-run／apply、備份還原、舊 flat-frame 的 caller fallback，以及改名改接線流程見 [Record device 升級指南](record-device-upgrades.md)。Record 範圍不改本頁的目前 component 版本或 lab 升級步驟。

## Lab-author 契約

`zcu_lab.components.parameter_upgrades` 宣告目前版本與 `parameter_upgrade_plan()`。`register_all` 將具名 `ParameterSchemaPolicy` 與 models 一起交給既有 ComponentRegistry。未配置 policy 時，框架拒絕建立或讀取 component 文件。

每個結構變更新增一個純函式 vN→vN+1 步驟。步驟接收具名 `ParameterDocument`，回傳含下一個嚴格整數 `schema_version` 的完整文件，不讀寫檔案。Runner 保留可選的 source_path 作診斷與容器種類判斷。Callback 不能從這個路徑讀檔。Caller 的 `validate(document, version)` 檢查該版完整容器與所有 components，包含半升級或不一致形狀。Runner 在每步前後驗證，current 文件也不能跳過檢查。

未標版文件只在 `classify_legacy` 完整確認舊格式後視為 v1。不能用「有一個舊欄位」取代完整分類。分類拒絕或無法確認時，回報原因並保持來源不變。

優先新增有預設值的欄位，不為新增 optional 欄位升版。結構步驟附真實舊文件樣本，與 [格式歷程](storage-format-history/parameter-schema-version.md) 同 commit 交付。通用機制的框架測試注入合成分類與步驟，lab 的真實樣本另做離線驗收。只有使用者確認所有資料都已升過某版後，才可刪除更早的步驟。

通用型別與失敗契約見 `zcu_tools.resources.parameter_upgrade` 的 docstring。版本邊界見 [ADR-0063](adr/0063-persistence-ownership.md#component-文件版本與明確升級)。長期入口不依賴一次性 `migrations/`，舊 repo 匯入仍使用 [storage migration](storage-migration.md)。
