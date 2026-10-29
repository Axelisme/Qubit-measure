# Component 參數文件升級

**讀者：** operator、lab-author、agent。辨識 component 文件版本，預覽與套用升級，並從備份還原。

## 版本與目前限制

`parameters.yaml`、每個 `points/<label>/parameters.yaml` 與 entry template 都有整份文件的 `schema_version: N`。它描述 component schema。`format` 與 `format_version` 仍描述 YAML 容器，兩種版本不互相代替。

GUI／MCP 只讀 lab 宣告的目前版本。缺版或不同版本會在讀取處報錯，列出來源路徑、文件版本與目前版本。`Preview:` 一行預覽差異，`Apply:` 一行含 `--apply`，備份後改檔。Setup 或 point 的兩行指令都以整個 entry 目錄為目標，entry template 則以該檔案為目標。指令會 quote 含空白的路徑，可直接複製執行。讀取不改檔。不要為了消除錯誤手改版本標頭。

目前 lab runtime 與升級目標都是 `schema_version: 3`，保留 flux v1→v2→v3 純函式步驟。V2 仍是每台儀器 half／integer／period 與 `flux_source`，v3 才使用 offset／period mapping 與 `flux_dev`。Runtime 只接受 v3，不做 legacy 相容讀取。

先在副本核對 dry-run 與備份，再安排正式資料升級。合法 v3 先完整驗證，再顯示 `skipped`。未標版文件只有通過完整 v1 分類後才視為 v1，不需要 operator 補標頭。Half 分支、缺名佔位、逐目的位置診斷、storage 來源與重校提醒見 [flux 升級指南](flux-migration.md)。

## 先改檔名，再升級內容

Runtime 只讀 entry 與 point 的 `parameters.yaml`、`library.yaml`、`aliases.yaml`。
舊 entry 的 `setup.yaml`、point 的 `point.yaml` 都改為 `parameters.yaml`。
兩層的 `module_cfg.yaml` 都改為 `library.yaml`，aliases 不變。
檔名變更不增加 schema_version 或容器版本。

先關閉會寫這些文件的工作階段，再預覽與套用改名：

```bash
uv run --directory /absolute/checkout --no-sync -- python scripts/upgrade_parameters.py \
  --scope filenames /data/project/result/device
uv run --directory /absolute/checkout --no-sync -- python scripts/upgrade_parameters.py \
  --scope filenames /data/project/result/device --apply
```

此 scope 接受 entry 目錄或精確舊檔路徑。Entry 目錄須含舊 setup 或新 parameters。
它只掃 entry 根與直屬 `points/*/`，不解析 YAML，也不升級 predictor。
Dry-run 列出舊路徑與新路徑，不建立檔案或鎖。
Apply 先保存原 bytes 備份，再建立新名稱並移除舊名稱，內容逐 byte 不變。
新舊並存或多個舊名指向同一新名時，整個選集在寫入前拒絕，不比較內容或覆寫。
重跑已改名的 entry 沒有待改項，回傳 0；已移走的精確舊檔路徑則回報缺檔。
Apply 中途 I/O 失敗可能留下已改名的前項，沒有多檔交易或掉電復原保證。
備份位置沿用下方共用規則。要還原名稱，先移走新名，再將備份複製回舊路徑。

需要內容升級時，先完成上述 `filenames`，再執行下方 `components`，預設 scope 仍為 components。
Components 只枚舉新檔名，不代做改名或讀取舊名。
改名歷程見 [文件檔名統一](storage-format-history/document-file-names.md)。

## 預覽與套用

從 checkout root 執行，使用該 worktree 的 interpreter。給 entry 目錄即可選取 `parameters.yaml`、所有 `points/*/parameters.yaml` 與 `predictors/*.json`，不必先開啟 entry 或補版本。目錄須含 `parameters.yaml`，否則 CLI 報錯。也可混合傳入多個 entry 目錄與明確 YAML 檔案路徑，或用 shell glob 選取檔案。

```bash
uv run --directory /absolute/checkout --no-sync -- python scripts/upgrade_parameters.py \
  /data/result/device
```

Predictor JSON 由獨立的 v1→v2 owner 驗證與轉換，不套用 component schema plan。報告列出 predictor 的版本、offset、period 與 unit。舊檔的佔位儀器名、單位預設與還原見 [Predictor v2](storage-format-history/predictor-v2.md)。

預設是 `dry-run`。輸出按完整檔案路徑列出結果，point 路徑含 label。每筆欄位差異顯示 literal key tuple、原值與新值，`<missing>` 與 explicit null 不同。Dry-run 不建檔、不建鎖、不備份。

先關閉會寫這些文件的工作階段，審閱差異後才加 `--apply`。

```bash
uv run --directory /absolute/checkout --no-sync -- python scripts/upgrade_parameters.py \
  /data/result/device --apply
```

Apply 先檢查備份位置，再取得 [per-user runtime lock](../lib/zcu_tools/resources/README.md#檔案鎖)，最多等待 5 秒。Runner 在鎖內重讀來源並驗證整條鏈，再保存完整原 bytes。備份成功才以同目錄 temporary 檔原子替換來源。輸出列出備份路徑與 `applied`，替換失敗也保留並列出已完成的備份。

### 備份位置

檔名、component、predictor、record 與 processed spectrum 升級共用備份規則。工具從來源向上找最近的 `result` 或 `Database` 目錄，其父目錄是資料專案根。備份位於 `<專案根>-backup/`，不是工具 checkout 旁，也不寫進資料項目。它鏡像來源相對於專案根的路徑，並加 UUID 防止覆寫。例如 `/data/project/result/device/parameters.yaml` 備份為 `/data/project-backup/result/device/parameters.yaml.<UUID>.bak`。此根目錄也供 storage migration 使用。

`--backup-root /absolute/archive-backup` 可覆寫備份根目錄。有專案根時仍鏡像相對路徑。來源不在 `result`／`Database` 下時，apply 必須提供此選項，備份直接放在 `<backup-root>/<原檔名>.<UUID>.bak`。例如升級 checkout 中的 entry template 時，指定 checkout 外的備份目錄。公開 runner 接受對應的 `backup_root=Path(...)`。

工具先解析備份根目錄的 symlink，再按解析後的名稱與直接子目錄檢查。若根目錄直接含 `result` 或 `Database` 子目錄，名稱必須以 `-backup` 結尾，否則拒絕並提示選 `*-backup` 目錄或空目錄。因此不能直接選另一個資料專案根作備份根。`*-backup` 根可接續既有鏡像，也可與 storage migration 備份共處。預設 `<專案根>-backup/` 符合此規則。檢查不讀檔案內容，也不建立標記檔。

不存在或沒有上述子目錄的根可沿用其他安全規則。未以 `-backup` 命名的空根首次使用後，若已產生 `result`／`Database` 鏡像，後續 apply 就會拒絕。要連續升級多份文件或接續備份，請一開始就選 `*-backup` 名稱。

備份根目錄仍不得位於任何 `result`／`Database` 資料樹內。Standalone 來源也拒絕位於來源父目錄內的備份。鏡像子目錄的 symlink 不得跳出備份根，也不得導入另一個 `result`／`Database` 資料樹。這些限制同樣適用於 `*-backup` 根，均在建立鎖或 temporary 前檢查。專案內其他位置可作覆寫目錄。Dry-run 不需要建立備份位置。

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
